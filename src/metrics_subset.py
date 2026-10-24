"""Point / Severe / exceedance metrics for every model column present in the
(minimal-run) preds files, using the metric definitions of evaluate.py.
Writes results/metrics_subset.csv and results/exceedance_subset.csv without
touching the archived full tables.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import HORIZONS, NON_MODEL, metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def main():
    rows, exc = [], []
    for h in HORIZONS:
        f = os.path.join(RES, "preds_h%d.parquet" % h)
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        te = df[df.split == "test"].copy()
        models = [c for c in te.columns if c not in NON_MODEL and te[c].dtype.kind == "f"]
        base = metrics(te.y, te.persistence)["mae"]
        for mod in models:
            r = metrics(te.y, te[mod])
            sv = te[te.y > 250]
            rs = metrics(sv.y, sv[mod])
            x0 = te.persistence
            esc = te[(x0 <= 250) & (te.y > 250)]
            re_ = metrics(esc.y, esc[mod])
            r.update(h=h, model=mod, skill_vs_pers=100 * (1 - r["mae"] / base),
                     mae_severe=rs["mae"], n_severe=rs["n"],
                     mae_escalation=re_["mae"], n_escalation=re_["n"])
            rows.append(r)
            for thr in (120, 250):
                m = te.y.notna() & te[mod].notna()
                obs = te.y[m] > thr
                pred = te[mod][m] > thr
                tp = int((obs & pred).sum()); fp = int((~obs & pred).sum())
                fn = int((obs & ~pred).sum())
                exc.append(dict(h=h, model=mod, thr=thr,
                                pod=tp / (tp + fn) if tp + fn else np.nan,
                                far=fp / (tp + fp) if tp + fp else np.nan,
                                csi=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                                n_events=int(obs.sum()), fp=fp))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "metrics_subset.csv"), index=False)
    pd.DataFrame(exc).to_csv(os.path.join(RES, "exceedance_subset.csv"), index=False)
    p = pd.DataFrame(rows)
    print(p.pivot(index="model", columns="h", values="mae").round(3))
    print(p.pivot(index="model", columns="h", values="mae_severe").round(2))
    e = pd.DataFrame(exc)
    print(e[e.thr == 250].pivot(index="model", columns="h", values="csi").round(3))


if __name__ == "__main__":
    main()
