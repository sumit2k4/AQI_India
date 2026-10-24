"""Statistical significance of model differences (revision, reviewer item 12.2).

For RGF (proposed column from results/proposed.json) against each competitor
in COMPETITORS, at every horizon, on the 2025 test year:

1. Diebold-Mariano test on the hourly network-mean loss differential
   d_t = mean_s(|e_A(s,t)|) - mean_s(|e_B(s,t)|)   (absolute-error loss),
   with a Bartlett-kernel HAC variance and bandwidth h + 24 hours
   (Diebold & Mariano 1995). Computed for all hours and for Severe hours
   (observed y > 250 ug/m3, averaging only over stations in that state).
2. Stationary block bootstrap (Politis & Romano 1994) over calendar time
   (mean block length 168 h, B = 1000 replicates, all stations kept together
   at every resampled timestamp) giving 95 % percentile intervals for the
   differences in MAE, Severe-hour MAE and CSI at the 250 ug/m3 threshold.

Output: results/significance.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
HORIZONS = [1, 6, 12, 24]
COMPETITORS = ["tcn", "lstm", "stpf", "stpf_w", "lstm2", "tcn2", "transformer"]
THR = 250.0
B = 1000
MEAN_BLOCK = 168
SEED = 42
try:
    PROPOSED = json.load(open(os.path.join(RES, "proposed.json")))["col"]
except Exception:
    PROPOSED = "rgdf_nodecomp"


def dm_test(d, bandwidth):
    """Diebold-Mariano statistic with Bartlett HAC variance."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    s = gamma0
    for k in range(1, bandwidth + 1):
        gk = np.dot(dc[k:], dc[:-k]) / n
        s += 2.0 * (1.0 - k / (bandwidth + 1.0)) * gk
    se = np.sqrt(max(s, 1e-12) / n)
    stat = dbar / se
    # Harvey-Leybourne-Newbold small-sample correction (negligible here)
    hln = np.sqrt((n + 1 - 2 * bandwidth + bandwidth * (bandwidth - 1) / n) / n)
    stat_c = stat * hln
    p = 2 * (1 - stats.t.cdf(abs(stat_c), df=n - 1))
    return dbar, stat_c, p, n


def stationary_bootstrap_idx(n, mean_block, rng):
    """Indices of one stationary-bootstrap resample of length n."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        L = rng.geometric(p)
        L = min(L, n - i)
        idx[i:i + L] = (start + np.arange(L)) % n
        i += L
    return idx


def per_time_sums(te, col):
    """Per-timestamp sufficient statistics for MAE, Severe MAE and CSI."""
    e = (te[col] - te.y).abs()
    sev = te.y > THR
    pred = te[col] > THR
    g = pd.DataFrame({
        "ts": te.ts, "ae": e, "n": 1.0,
        "ae_sev": e.where(sev, 0.0), "n_sev": sev.astype(float),
        "tp": (sev & pred).astype(float), "fp": (~sev & pred).astype(float),
        "fn": (sev & ~pred).astype(float)}).groupby("ts").sum()
    return g


def metrics_from_sums(S, idx=None):
    s = S if idx is None else S.iloc[idx]
    tot = s.sum()
    mae = tot.ae / tot.n
    mae_sev = tot.ae_sev / tot.n_sev
    csi = tot.tp / (tot.tp + tot.fp + tot.fn)
    return mae, mae_sev, csi


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for h in HORIZONS:
        fh = os.path.join(RES, "preds_h%d.parquet" % h)
        if not os.path.exists(fh):
            continue
        df = pd.read_parquet(fh)
        if PROPOSED not in df.columns:
            continue
        te = df[df.split == "test"].copy()
        te = te.sort_values(["ts", "sid"])
        for comp in COMPETITORS:
            if comp not in te.columns:
                continue
            m = te.y.notna() & te[PROPOSED].notna() & te[comp].notna()
            t = te[m]
            SA = per_time_sums(t, PROPOSED)
            SB = per_time_sums(t, comp)
            SA, SB = SA.align(SB, join="inner")
            # ---- DM on network-mean loss differential
            dA = SA.ae / SA.n
            dB = SB.ae / SB.n
            dbar, stat, p, n = dm_test((dA - dB).values, h + 24)
            msev = SA.n_sev > 0
            dAs = (SA.ae_sev / SA.n_sev)[msev]
            dBs = (SB.ae_sev / SB.n_sev)[msev]
            dbar_s, stat_s, p_s, n_s = dm_test((dAs - dBs).values, h + 24)
            # ---- block bootstrap on differences
            base = np.array(metrics_from_sums(SA)) - np.array(metrics_from_sums(SB))
            boots = np.empty((B, 3))
            nT = len(SA)
            for b in range(B):
                idx = stationary_bootstrap_idx(nT, MEAN_BLOCK, rng)
                boots[b] = (np.array(metrics_from_sums(SA, idx))
                            - np.array(metrics_from_sums(SB, idx)))
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            rows.append(dict(
                h=h, proposed=PROPOSED, competitor=comp,
                d_mae=base[0], d_mae_lo=lo[0], d_mae_hi=hi[0],
                dm_stat_all=stat, dm_p_all=p, n_dm_all=n,
                d_mae_sev=base[1], d_mae_sev_lo=lo[1], d_mae_sev_hi=hi[1],
                dm_stat_sev=stat_s, dm_p_sev=p_s, n_dm_sev=n_s,
                d_csi=base[2], d_csi_lo=lo[2], d_csi_hi=hi[2]))
            print("h=%d %s-%s dMAE %.2f [%.2f,%.2f] p=%.3g | dSev %.1f [%.1f,%.1f] "
                  "p=%.3g | dCSI %.3f [%.3f,%.3f]" % (
                      h, PROPOSED, comp, base[0], lo[0], hi[0], p, base[1], lo[1],
                      hi[1], p_s, base[2], lo[2], hi[2]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "significance.csv"), index=False)
    print("saved significance.csv")


if __name__ == "__main__":
    main()
