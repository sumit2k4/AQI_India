"""Severity-weighted STPF variant (STPF-W): identical features and
hyperparameters to STPF, but training samples weighted w = 1 + (y/250)^2
(capped at 20), anchoring the ramp at the Indian AQI Severe breakpoint.
The scheme is fixed a priori - no tuning on calibration or test data.

Appends column 'stpf_w' to results/preds_h{h}.parquet.
"""
import gc
import os
import sys

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")


def main():
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    T.meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = T.build_base(df)
    for h in T.HORIZONS:
        f_check = os.path.join(RES, "preds_h%d.parquet" % h)
        import pyarrow.parquet as pq
        if "stpf_w" in pq.read_schema(f_check).names:
            print("horizon", h, "already has stpf_w, skipping", flush=True)
            continue
        print("=== weighted, horizon", h, "===", flush=True)
        data = T.make_features(panels, full_idx, h)
        tr = data[(data.ts >= "2019-01-08")
                  & (data.ts <= pd.Timestamp("2023-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        feat_all = [c for c in data.columns if c not in ("y", "ts", "sid", "city")]
        ct = data[(data.ts >= "2024-01-01") & (data.ts <= "2026-01-01")]
        del data  # memory: tr and ct are copies
        gc.collect()
        w = (1.0 + (tr.y / 250.0) ** 2).clip(upper=20.0)
        m = lgb.LGBMRegressor(**T.LGB_PARAMS)
        m.fit(tr[feat_all], tr.y, sample_weight=w,
              categorical_feature=["sid_code"])
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        preds = pd.read_parquet(f)
        pieces = []
        for split, lo, hi in (("cal", "2024-01-01", "2024-12-31"),
                              ("test", "2025-01-01", "2026-01-01")):
            d = ct[(ct.ts >= lo) & (ct.ts <= hi)]
            p = d[["ts", "sid"]].copy()
            p["stpf_w"] = m.predict(d[feat_all]).astype("float32")
            pieces.append(p)
        newcol = pd.concat(pieces, ignore_index=True)
        if "stpf_w" in preds.columns:
            preds = preds.drop(columns=["stpf_w"])
        preds = preds.merge(newcol, on=["ts", "sid"], how="left")
        preds.to_parquet(f, index=False)
        print("appended stpf_w to", f, flush=True)
        del tr, ct
        gc.collect()
    print("WEIGHTED DONE", flush=True)


if __name__ == "__main__":
    main()
