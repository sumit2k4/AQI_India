"""Compare rerun test-year metrics with the archived submission tables.

    python src/repro_check.py 24          # one horizon
Prints overall and Severe-hour MAE for every model column present in
results/preds_h{h}.parquet next to results/archive_oct22/{point,episode}_metrics.csv.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ARC = os.path.join(RES, "archive_oct22")
SKIP = {"ts", "sid", "city", "y", "split"}


def main(h):
    df = pd.read_parquet(os.path.join(RES, "preds_h%d.parquet" % h))
    te = df[df.split == "test"]
    pm = pd.read_csv(os.path.join(ARC, "point_metrics.csv"))
    ep = pd.read_csv(os.path.join(ARC, "episode_metrics.csv"))
    for c in [c for c in te.columns if c not in SKIP and te[c].dtype.kind == "f"]:
        m = te.y.notna() & te[c].notna()
        mae = (te[c][m] - te.y[m]).abs().mean()
        s = m & (te.y > 250)
        sev = (te[c][s] - te.y[s]).abs().mean()
        a = pm[(pm.h == h) & (pm.model == c)]
        b = ep[(ep.h == h) & (ep.model == c) & (ep.subset == "severe")]
        am = a.mae.iloc[0] if len(a) else float("nan")
        bs = b.mae.iloc[0] if len(b) else float("nan")
        print("h=%d %-15s MAE %.3f (archive %.3f, n %d)  Severe %.2f (archive %.2f)"
              % (h, c, mae, am, m.sum(), sev, bs), flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:] or ["24"]:
        main(int(a))
