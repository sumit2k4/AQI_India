"""Minimal prediction set for the revision analyses (significance, sensitivity,
re-tuned deep baselines). Uses exactly the feature builder, LightGBM
parameters, severity weights and fuzzy memberships of train_eval.py and
rgdf.py, but fits only the columns the new tables need:

  h = 1, 6, 12, 24 : y, split, persistence             (targets for the DL rows)
  h = 6, 24        : stpf (unweighted fusion GBM), rgdf_nodecomp (RGF)

Writes results/preds_h{h}.parquet (same layout as the full pipeline, fewer
columns). The archived full result tables in results/ are not touched.
"""
import gc
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as T  # noqa: E402
from rgdf import REGIMES, memberships  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
FIT_H = (6, 24)


def main():
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    T.meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = T.build_base(df)
    del df
    for h in T.HORIZONS:
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        if os.path.exists(f):
            print("horizon", h, "exists, skipping", flush=True)
            continue
        t0 = time.time()
        print("=== horizon", h, "===", flush=True)
        data = T.make_features(panels, full_idx, h)
        tr = data[(data.ts >= "2019-01-08")
                  & (data.ts <= pd.Timestamp("2023-12-31 23:00") - pd.Timedelta(hours=h))]
        ca = data[(data.ts >= "2024-01-01")
                  & (data.ts <= pd.Timestamp("2024-12-31 23:00") - pd.Timedelta(hours=h))]
        te = data[(data.ts >= "2025-01-01")]
        feat_all = [c for c in data.columns if c not in ("y", "ts", "sid", "city")]
        del data
        gc.collect()
        print("rows train/cal/test:", len(tr), len(ca), len(te), flush=True)
        out = {}
        for split, d in (("cal", ca), ("test", te)):
            r = d[["ts", "sid", "city", "y"]].copy()
            r["persistence"] = d["pm25_lag0"].values
            out[split] = r
        if h in FIT_H:
            Xtr, ytr = tr[feat_all], tr["y"]
            m_stpf = lgb.LGBMRegressor(**T.LGB_PARAMS)
            m_stpf.fit(Xtr, ytr, categorical_feature=["sid_code"])
            for split, d in (("cal", ca), ("test", te)):
                out[split]["stpf"] = m_stpf.predict(d[feat_all]).astype("float32")
            del m_stpf
            print("stpf done %.0f s" % (time.time() - t0), flush=True)
            # RGF: gated severity-weighted experts on y (rgdf.py "nodecomp")
            wl_tr, wm_tr, wh_tr = memberships(tr.pm25_lag0)
            Wtr = dict(low=wl_tr, mid=wm_tr, high=wh_tr)
            sev_w = np.minimum(1.0 + (tr.y.values / 250.0) ** 2, 20.0)
            frames = (ca, te)
            Wfr = []
            for d in frames:
                wl, wm, wh = memberships(d.pm25_lag0)
                Wfr.append(dict(low=wl, mid=wm, high=wh))
            Ydir = [np.zeros(len(d), dtype="float64") for d in frames]
            for r in REGIMES:
                m = Wtr[r] > 0
                mdl = lgb.LGBMRegressor(**T.LGB_PARAMS)
                mdl.fit(tr.loc[m, feat_all], tr.loc[m, "y"],
                        sample_weight=(Wtr[r] * sev_w)[m],
                        categorical_feature=["sid_code"])
                for i, d in enumerate(frames):
                    Ydir[i] += Wfr[i][r] * mdl.predict(d[feat_all]).astype("float32")
                del mdl
                gc.collect()
                print("  expert", r, "done %.0f s" % (time.time() - t0), flush=True)
            out["cal"]["rgdf_nodecomp"] = Ydir[0].astype("float32")
            out["test"]["rgdf_nodecomp"] = Ydir[1].astype("float32")
        res = pd.concat([out["cal"].assign(split="cal"),
                         out["test"].assign(split="test")], ignore_index=True)
        res.to_parquet(f, index=False)
        te_r = res[res.split == "test"]
        for c in ("persistence", "stpf", "rgdf_nodecomp"):
            if c in te_r.columns:
                mm = te_r.y.notna() & te_r[c].notna()
                print("  test MAE %s: %.3f (n=%d)" % (c, (te_r[c][mm] - te_r.y[mm]).abs().mean(),
                                                     mm.sum()), flush=True)
        print("saved", f, "%.0f s" % (time.time() - t0), flush=True)
        del tr, ca, te, res, out
        gc.collect()
    print("MINIMAL DONE", flush=True)


if __name__ == "__main__":
    main()
