"""Sensitivity of RGF (gated severity-weighted experts, direct form) at h = 24
to the three design constants that are not AQI breakpoints (revision,
reviewer item 12.4):

  band scale   0.5 / 1 / 2   -> half-widths (+-5,+-15) / (+-10,+-30) / (+-20,+-60)
  ramp power   1 / 2 / 3     -> w = min(1 + (y/250)^p, cap)
  cap          10 / 20 / inf

One constant is varied at a time around the default (1, 2, 20). Every variant
is trained on 2019-2023 exactly like rgdf.py (no calibration-year tuning) and
evaluated once on the 2025 test year: MAE all, Severe MAE (y > 250),
CSI at 250 and escalation MAE (issue <= 250, target > 250).

Output: results/sensitivity_h24.csv
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
H = 24
REGIMES = ("low", "mid", "high")
VARIANTS = [  # (name, band_scale, power, cap)
    ("default", 1.0, 2.0, 20.0),
    ("band_x0.5", 0.5, 2.0, 20.0),
    ("band_x2", 2.0, 2.0, 20.0),
    ("power_1", 1.0, 1.0, 20.0),
    ("power_3", 1.0, 3.0, 20.0),
    ("cap_10", 1.0, 2.0, 10.0),
    ("cap_none", 1.0, 2.0, np.inf),
]


def memberships(x, scale=1.0):
    x = np.asarray(x, dtype="float64")
    a1, a2 = 10.0 * scale, 30.0 * scale          # half-widths at 60 and 250
    w_low = np.clip((60.0 + a1 - x) / (2 * a1), 0.0, 1.0)
    w_high = np.clip((x - (250.0 - a2)) / (2 * a2), 0.0, 1.0)
    w_mid = np.clip(1.0 - w_low - w_high, 0.0, 1.0)
    tot = w_low + w_mid + w_high
    return dict(low=w_low / tot, mid=w_mid / tot, high=w_high / tot)


def main():
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    T.meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = T.build_base(df)
    data = T.make_features(panels, full_idx, H)
    tr = data[(data.ts >= "2019-01-08")
              & (data.ts <= pd.Timestamp("2023-12-31 23:00") - pd.Timedelta(hours=H))]
    te = data[data.ts >= "2025-01-01"]
    feat = [c for c in data.columns if c not in ("y", "ts", "sid", "city")]
    del data  # memory: tr/te are copies
    gc.collect()
    y_te = te.y.values
    x0_te = te.pm25_lag0.values
    rows = []
    for name, scale, power, cap in VARIANTS:
        print("=== variant", name, flush=True)
        Wtr = memberships(tr.pm25_lag0, scale)
        Wte = memberships(te.pm25_lag0, scale)
        sev = np.minimum(1.0 + (tr.y.values / 250.0) ** power, cap)
        pred = np.zeros(len(te), dtype="float64")
        for r in REGIMES:
            m = Wtr[r] > 0
            mdl = lgb.LGBMRegressor(**T.LGB_PARAMS)
            mdl.fit(tr.loc[m, feat], tr.loc[m, "y"], sample_weight=(Wtr[r] * sev)[m],
                    categorical_feature=["sid_code"])
            pred += Wte[r] * mdl.predict(te[feat])
            del mdl
            gc.collect()
        ok = ~np.isnan(y_te)
        e = np.abs(pred - y_te)
        sevm = ok & (y_te > 250)
        esc = ok & (x0_te <= 250) & (y_te > 250)
        obs = y_te[ok] > 250
        prd = pred[ok] > 250
        tp = int((obs & prd).sum()); fp = int((~obs & prd).sum()); fn = int((obs & ~prd).sum())
        rows.append(dict(variant=name, band_scale=scale, power=power, cap=cap,
                         mae_all=float(e[ok].mean()), mae_severe=float(e[sevm].mean()),
                         csi_250=tp / (tp + fp + fn), pod_250=tp / (tp + fn),
                         far_250=fp / (tp + fp), mae_escalation=float(e[esc].mean()),
                         n=int(ok.sum())))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "sensitivity_h24.csv"), index=False)
    print("SENSITIVITY DONE")


if __name__ == "__main__":
    main()
