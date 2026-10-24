"""Conformal calibration of the RGDF regime-gated quantile heads.

Methods compared (target 90% coverage), calibrated on 2024, tested on 2025:
  raw   uncalibrated gated quantile interval [q05, q95]
  scp   split conformal: symmetric band around the RGDF point forecast
  cqr   conformalized quantile regression, marginal correction
  mond  regime-conditional (Mondrian) CQR: correction per issue-time regime
        (Indian AQI PM2.5 bins <=60, 60-120, 120-250, >250 ug/m3)

Outputs: results/intervals_h{h}.parquet (point + all four interval variants),
         results/conformal_summary.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
HORIZONS = [1, 6, 12, 24]
ALPHA = 0.10
BINS = [-np.inf, 60, 120, 250, np.inf]
BIN_NAMES = ["<=60", "60-120", "120-250", ">250"]
QLO, QHI = "rgdf_q05", "rgdf_q95"
try:
    import json
    POINT = json.load(open(os.path.join(RES, "proposed.json")))["col"]
except Exception:
    POINT = "rgdf_nodecomp"


def qhat(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores)[k - 1]


def main():
    rows = []
    for h in HORIZONS:
        df = pd.read_parquet(os.path.join(RES, "preds_h%d.parquet" % h))
        df["regime"] = pd.cut(df["persistence"], BINS, labels=BIN_NAMES)
        cal = df[df.split == "cal"].dropna(subset=["y", QLO, QHI, POINT])
        te = df[df.split == "test"].dropna(subset=["y", QLO, QHI, POINT]).copy()

        te["lo_raw"], te["hi_raw"] = te[QLO], te[QHI]

        r = np.abs(cal.y - cal[POINT])
        q_scp = qhat(r.values, ALPHA)
        te["lo_scp"], te["hi_scp"] = te[POINT] - q_scp, te[POINT] + q_scp

        e = np.maximum(cal[QLO] - cal.y, cal.y - cal[QHI])
        q_cqr = qhat(e.values, ALPHA)
        te["lo_cqr"], te["hi_cqr"] = te[QLO] - q_cqr, te[QHI] + q_cqr

        te["lo_mond"], te["hi_mond"] = np.nan, np.nan
        for b in BIN_NAMES:
            cb = cal[cal.regime == b]
            tb = te.regime == b
            if len(cb) < 50:
                qb = q_cqr
            else:
                eb = np.maximum(cb[QLO] - cb.y, cb.y - cb[QHI])
                qb = qhat(eb.values, ALPHA)
            te.loc[tb, "lo_mond"] = te.loc[tb, QLO] - qb
            te.loc[tb, "hi_mond"] = te.loc[tb, QHI] + qb

        keep = ["ts", "sid", "city", "y", "regime", POINT,
                "lo_raw", "hi_raw", "lo_scp", "hi_scp",
                "lo_cqr", "hi_cqr", "lo_mond", "hi_mond"]
        te[keep].rename(columns={POINT: "point"}).to_parquet(
            os.path.join(RES, "intervals_h%d.parquet" % h), index=False)

        for meth in ["raw", "scp", "cqr", "mond"]:
            lo = te["lo_" + meth].clip(lower=0)
            hi = te["hi_" + meth]
            cover = (te.y >= lo) & (te.y <= hi)
            width = hi - lo
            rows.append({"h": h, "method": meth, "group": "overall",
                         "coverage": cover.mean() * 100,
                         "width": width.median(), "n": len(te)})
            for b in BIN_NAMES:
                m = te.regime == b
                if m.sum():
                    rows.append({"h": h, "method": meth, "group": b,
                                 "coverage": cover[m].mean() * 100,
                                 "width": width[m].median(),
                                 "n": int(m.sum())})
        print("h=%d q_scp=%.1f q_cqr=%.1f" % (h, q_scp, q_cqr), flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RES, "conformal_summary.csv"),
                              index=False)
    print("saved conformal_summary.csv")


if __name__ == "__main__":
    main()
