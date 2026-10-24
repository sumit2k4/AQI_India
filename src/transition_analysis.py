"""Hard vs fuzzy gating analysis near regime transition zones.

Trains a hard-gated arm (identical experts, features and severity weights;
memberships replaced by hard indicators at 60 and 250 ug/m3, predictions
routed by the issue-time hard regime) and appends 'rgdf_hard' to the preds
files. Then computes, on the 2025 test year:

1. Band-conditional MAE/RMSE by issue-time concentration x_t:
   floor [0,10) (instrument detection floor, reported for completeness),
   core-low [10,50), transition-1 [50,70], core-mid (70,220),
   transition-2 [220,280], core-high (280, inf)
   for hard gate, fuzzy gate (proposed), single severity-weighted GBM and
   unweighted GBM -> results/transition_metrics.csv

2. Prediction-continuity analysis: mean absolute change of consecutive-hour
   forecasts |yhat_t - yhat_{t-1}| for issue pairs that cross a regime
   boundary vs pairs that do not, with observed target change as reference
   -> results/transition_jumps.csv

3. Fine-bin MAE curve over x_t (20 ug/m3 bins) -> results/transition_curve.csv
"""
import gc
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

BOUNDS = (60.0, 250.0)
BANDS = [("floor", 0, 10), ("core-low", 10, 50), ("transition-1", 50, 70),
         ("core-mid", 70, 220), ("transition-2", 220, 280),
         ("core-high", 280, np.inf)]
MODELS = {"rgdf_hard": "Hard gate", "rgdf_nodecomp": "Fuzzy gate (RGF)",
          "stpf_w": "Single GBM, weighted", "stpf": "Single GBM, unweighted"}


def hard_regime(x):
    x = np.asarray(x, dtype="float64")
    r = np.ones(len(x), dtype=int)          # mid
    r[x < BOUNDS[0]] = 0
    r[x > BOUNDS[1]] = 2
    return r


def train_hard():
    import pyarrow.parquet as pq
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    T.meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = T.build_base(df)
    for h in T.HORIZONS:
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        if "rgdf_hard" in pq.read_schema(f).names:
            print("h%d already has rgdf_hard, skipping" % h, flush=True)
            continue
        print("=== hard gate, horizon", h, "===", flush=True)
        data = T.make_features(panels, full_idx, h)
        tr = data[(data.ts >= "2019-01-08")
                  & (data.ts <= pd.Timestamp("2023-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        ca = data[(data.ts >= "2024-01-01")
                  & (data.ts <= pd.Timestamp("2024-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        te = data[data.ts >= "2025-01-01"]
        feat = [c for c in data.columns if c not in
                ("y", "ts", "sid", "city")]
        del data  # memory: slices are copies
        gc.collect()
        sev = np.minimum(1.0 + (tr.y.values / 250.0) ** 2, 20.0)
        rtr = hard_regime(tr.pm25_lag0)
        outs = {}
        for split, d in (("cal", ca), ("test", te)):
            outs[split] = np.full(len(d), np.nan, dtype="float64")
        for r in (0, 1, 2):
            m = rtr == r
            mdl = lgb.LGBMRegressor(**T.LGB_PARAMS)
            mdl.fit(tr.loc[m, feat], tr.loc[m, "y"], sample_weight=sev[m],
                    categorical_feature=["sid_code"])
            for split, d in (("cal", ca), ("test", te)):
                mask = hard_regime(d.pm25_lag0) == r
                if mask.sum():
                    outs[split][mask] = mdl.predict(d.loc[mask, feat])
            print("  expert", r, "n_train=%d" % m.sum(), flush=True)
        preds = pd.read_parquet(f)
        pieces = []
        for split, d in (("cal", ca), ("test", te)):
            p = d[["ts", "sid"]].copy()
            p["rgdf_hard"] = outs[split].astype("float32")
            pieces.append(p)
        newcol = pd.concat(pieces, ignore_index=True)
        if "rgdf_hard" in preds.columns:
            preds = preds.drop(columns=["rgdf_hard"])
        preds = preds.merge(newcol, on=["ts", "sid"], how="left")
        preds.to_parquet(f, index=False)
        print("appended rgdf_hard to", f, flush=True)
        del tr, ca, te
        gc.collect()


def analyse():
    rows, jrows, crows = [], [], []
    for h in T.HORIZONS:
        df = pd.read_parquet(os.path.join(RES, "preds_h%d.parquet" % h))
        te = df[df.split == "test"].copy()
        x = te.persistence
        for name in MODELS:
            ok = te.y.notna() & te[name].notna()
            for band, lo, hi in BANDS:
                m = ok & (x >= lo) & (x < hi if np.isfinite(hi)
                                      else x >= lo)
                if np.isfinite(hi):
                    m = ok & (x >= lo) & (x <= hi)
                e = te.loc[m, name] - te.loc[m, "y"]
                rows.append(dict(h=h, model=name, band=band,
                                 mae=float(e.abs().mean()),
                                 rmse=float(np.sqrt((e ** 2).mean())),
                                 n=int(m.sum())))
            # fine curve (h=24 only, 20 ug bins to 400)
            if h == 24:
                for lo in range(0, 400, 20):
                    m = ok & (x >= lo) & (x < lo + 20)
                    if m.sum() < 200:
                        continue
                    e = te.loc[m, name] - te.loc[m, "y"]
                    crows.append(dict(model=name, x_lo=lo,
                                      mae=float(e.abs().mean()),
                                      n=int(m.sum())))
        # continuity: consecutive-hour forecast change
        te = te.sort_values(["sid", "ts"])
        same = (te.sid == te.sid.shift(1)) & \
               ((te.ts - te.ts.shift(1)) == pd.Timedelta("1H"))
        x0, x1 = te.persistence.shift(1), te.persistence
        cross = same & (((x0 - BOUNDS[0]) * (x1 - BOUNDS[0]) < 0)
                        | ((x0 - BOUNDS[1]) * (x1 - BOUNDS[1]) < 0))
        nocross = same & ~cross
        dy = (te.y - te.y.shift(1)).abs()
        for name in MODELS:
            dp = (te[name] - te[name].shift(1)).abs()
            jrows.append(dict(
                h=h, model=name,
                jump_cross=float(dp[cross].mean()),
                jump_nocross=float(dp[nocross].mean()),
                ratio=float(dp[cross].mean() / dp[nocross].mean()),
                n_cross=int(cross.sum())))
        jrows.append(dict(h=h, model="observed_target",
                          jump_cross=float(dy[cross].mean()),
                          jump_nocross=float(dy[nocross].mean()),
                          ratio=float(dy[cross].mean() / dy[nocross].mean()),
                          n_cross=int(cross.sum())))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "transition_metrics.csv"),
                              index=False)
    pd.DataFrame(jrows).to_csv(os.path.join(RES, "transition_jumps.csv"),
                               index=False)
    pd.DataFrame(crows).to_csv(os.path.join(RES, "transition_curve.csv"),
                               index=False)
    t = pd.DataFrame(rows)
    print(t[(t.h == 24) & t.band.str.startswith("transition")]
          .pivot(index="model", columns="band", values="mae").round(2))


if __name__ == "__main__":
    train_hard()
    analyse()
    print("TRANSITION DONE", flush=True)
