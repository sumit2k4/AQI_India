"""Fold-2 RGF stage: gated severity-weighted experts on y (the configuration reported
as RGF) and the gated q05/q95 quantile heads. The decomposition variants are not rebuilt.
Appends rgdf_nodecomp, rgdf_q05, rgdf_q95 to results/preds_h{h}.parquet."""
import gc
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as T
from rgdf import memberships, fit_predict, fit_predict_q, REGIMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")


def main():
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    T.meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = T.build_base(df)
    costf = os.path.join(RES, "cost_rgdf.csv")
    cost = pd.read_csv(costf).to_dict("records") if os.path.exists(costf) else []
    for h in T.HORIZONS:
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        if "rgdf_nodecomp" in pq.read_schema(f).names:
            print("horizon", h, "already has rgdf_nodecomp, skipping", flush=True)
            continue
        print("=== RGF fold2 horizon", h, "===", flush=True)
        data = T.make_features(panels, full_idx, h)
        tr = data[(data.ts >= "2019-01-08")
                  & (data.ts <= pd.Timestamp("2022-12-31 23:00") - pd.Timedelta(hours=h))]
        ca = data[(data.ts >= "2023-01-01")
                  & (data.ts <= pd.Timestamp("2023-12-31 23:00") - pd.Timedelta(hours=h))]
        te = data[(data.ts >= "2024-01-01")
                  & (data.ts <= pd.Timestamp("2024-12-31 23:00") - pd.Timedelta(hours=h))]
        feat = [c for c in data.columns if c not in ("y", "ts", "sid", "city")]
        frames = (ca, te)
        del data
        gc.collect()
        wl, wm, wh = memberships(tr.pm25_lag0)
        Wtr = dict(low=wl, mid=wm, high=wh)
        sev_w = np.minimum(1.0 + (tr.y.values / 250.0) ** 2, 20.0)
        Wfr = []
        for d in frames:
            a, b, c = memberships(d.pm25_lag0)
            Wfr.append(dict(low=a, mid=b, high=c))
        Ydir = [np.zeros(len(d), dtype="float64") for d in frames]
        for r in REGIMES:
            m = Wtr[r] > 0
            outs = fit_predict(tr.loc[m, feat], tr.loc[m, "y"], (Wtr[r] * sev_w)[m], frames, feat,
                               "rgdf_nodecomp_%s_h%d" % (r, h), cost)
            for i in range(len(frames)):
                Ydir[i] += Wfr[i][r] * outs[i]
        Q = {}
        for qv, qtag in ((0.05, "q05"), (0.95, "q95")):
            acc = [np.zeros(len(d), dtype="float64") for d in frames]
            for r in REGIMES:
                m = Wtr[r] > 0
                outs = fit_predict_q(tr.loc[m, feat], tr.loc[m, "y"], Wtr[r][m], frames, feat, qv,
                                     "rgdf_%s_%s_h%d" % (qtag, r, h), cost)
                for i in range(len(frames)):
                    acc[i] += Wfr[i][r] * outs[i]
            Q[qtag] = acc
        preds = pd.read_parquet(f)
        pieces = []
        for i, d in enumerate(frames):
            p = d[["ts", "sid"]].copy()
            p["rgdf_nodecomp"] = Ydir[i].astype("float32")
            p["rgdf_q05"] = Q["q05"][i].astype("float32")
            p["rgdf_q95"] = Q["q95"][i].astype("float32")
            pieces.append(p)
        newdf = pd.concat(pieces, ignore_index=True)
        preds = preds.drop(columns=[c for c in ("rgdf_nodecomp", "rgdf_q05", "rgdf_q95") if c in preds.columns])
        preds = preds.merge(newdf, on=["ts", "sid"], how="left")
        preds.to_parquet(f, index=False)
        pd.DataFrame(cost).to_csv(costf, index=False)
        print("appended RGF columns to", f, flush=True)
        del tr, ca, te
        gc.collect()
    print("RGDF DONE", flush=True)


if __name__ == "__main__":
    main()
