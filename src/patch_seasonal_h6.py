"""Fill the missing seasonal_naive column for h=6 in preds_h6.parquet.

seasonal_naive for horizon h is pm25(t + h - 24), i.e. lag (24-h) = 18 h,
which was not in the trained lag set. Computed exactly from the QC'd hourly
panel; no model involved.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

panel = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"),
                        columns=["ts", "sid", "pm25"])
lag18 = panel.copy()
lag18["ts"] = lag18.ts + pd.Timedelta(hours=18)
lag18 = lag18.rename(columns={"pm25": "sn6"})

f = os.path.join(RES, "preds_h6.parquet")
preds = pd.read_parquet(f)
before = preds.seasonal_naive.notna().sum()
preds = preds.merge(lag18, on=["ts", "sid"], how="left")
preds["seasonal_naive"] = preds["sn6"].astype("float32")
preds = preds.drop(columns=["sn6"])
preds.to_parquet(f, index=False)
print("seasonal_naive non-null: before=%d after=%d rows=%d"
      % (before, preds.seasonal_naive.notna().sum(), len(preds)))
