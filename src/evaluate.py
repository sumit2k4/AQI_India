"""Point-forecast evaluation v2: accuracy, episode, exceedance, seasonal and
regime-conditional metrics for every model column present in the preds files.

Outputs (all computed; injected verbatim into the manuscript):
  results/point_metrics.csv        MAE/RMSE/bias/R2/SMAPE/skill per model x h
  results/point_metrics_city.csv   per-city MAE
  results/episode_metrics.csv      MAE on y>120 and y>250 subsets
  results/exceedance_metrics.csv   POD/FAR/CSI at 120/250
  results/season_metrics.csv       MAE by season (DJF/MAM/JJA/SON)
  results/regime_metrics.csv       MAE by issue-time regime bin
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
HORIZONS = [1, 6, 12, 24]
NON_MODEL = {"ts", "sid", "city", "y", "split", "stpf_q05", "stpf_q50",
             "stpf_q95", "rgdf_q05", "rgdf_q95", "regime"}
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
RBINS = [-np.inf, 60, 120, 250, np.inf]
RNAMES = ["<=60", "60-120", "120-250", ">250"]


def metrics(y, p):
    m = y.notna() & p.notna()
    if m.sum() == 0:
        return dict(mae=np.nan, rmse=np.nan, bias=np.nan, r2=np.nan,
                    smape=np.nan, n=0)
    yy, pp = y[m].astype(float), p[m].astype(float)
    e = pp - yy
    sst = ((yy - yy.mean()) ** 2).sum()
    return dict(
        mae=float(np.abs(e).mean()),
        rmse=float(np.sqrt((e ** 2).mean())),
        bias=float(e.mean()),
        r2=float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan,
        smape=float((2 * np.abs(e) / (np.abs(yy) + np.abs(pp))).mean() * 100),
        n=int(m.sum()))


def select_proposed():
    """Choose the deployed framework configuration on the CALIBRATION year
    only (2024): combined normalized score of overall and severe MAE summed
    over horizons. Writes results/proposed.json."""
    import json
    cands = ["rgdf", "rgdf_nogate", "rgdf_nodecomp", "stpf_w"]
    scores = {c: 0.0 for c in cands}
    for h in HORIZONS:
        df = pd.read_parquet(os.path.join(RES, "preds_h%d.parquet" % h))
        ca = df[df.split == "cal"]
        present = [c for c in cands if c in ca.columns]
        m_all, m_sev = {}, {}
        for c in present:
            m = ca.y.notna() & ca[c].notna()
            m_all[c] = float((ca[c][m] - ca.y[m]).abs().mean())
            sv = ca[ca.y > 250]
            ms = sv.y.notna() & sv[c].notna()
            m_sev[c] = float((sv[c][ms] - sv.y[ms]).abs().mean())
        for c in present:
            scores[c] += (m_all[c] / min(m_all.values())
                          + m_sev[c] / min(m_sev.values()))
    sel = min(scores, key=scores.get)
    with open(os.path.join(RES, "proposed.json"), "w") as f:
        json.dump({"col": sel, "cal_scores": scores}, f, indent=1)
    print("proposed configuration (selected on 2024):", sel, scores)
    return sel


def main():
    select_proposed()
    point, city_rows, epi, exc, seas, regi = [], [], [], [], [], []
    for h in HORIZONS:
        df = pd.read_parquet(os.path.join(RES, "preds_h%d.parquet" % h))
        te = df[df.split == "test"].copy()
        te["season"] = te.ts.dt.month.map(SEASON)
        te["regime"] = pd.cut(te.persistence, RBINS, labels=RNAMES)
        models = [c for c in te.columns
                  if c not in NON_MODEL and c != "season"
                  and te[c].dtype.kind == "f"]
        base_mae = metrics(te.y, te.persistence)["mae"]
        for mod in models:
            r = metrics(te.y, te[mod])
            r.update(h=h, model=mod,
                     skill_vs_pers=100 * (1 - r["mae"] / base_mae))
            point.append(r)
            for city, g in te.groupby("city"):
                city_rows.append(dict(h=h, model=mod, city=city,
                                      **metrics(g.y, g[mod])))
            for sn, g in te.groupby("season"):
                seas.append(dict(h=h, model=mod, season=sn,
                                 mae=metrics(g.y, g[mod])["mae"],
                                 n=len(g)))
            for rb, g in te.groupby("regime"):
                regi.append(dict(h=h, model=mod, regime=rb,
                                 mae=metrics(g.y, g[mod])["mae"],
                                 n=len(g)))
            # transition subsets: escalation (begin <=250, end >250),
            # decay (begin >250, end <=250), sustained (both >250)
            x0 = te.persistence
            for tag, msk in (
                    ("escalation", (x0 <= 250) & (te.y > 250)),
                    ("decay", (x0 > 250) & (te.y <= 250)),
                    ("sustained", (x0 > 250) & (te.y > 250))):
                sub = te[msk]
                regi.append(dict(h=h, model=mod, regime=tag,
                                 mae=metrics(sub.y, sub[mod])["mae"],
                                 n=len(sub)))
            for thr, tag in ((120, "vpoor"), (250, "severe")):
                sub = te[te.y > thr]
                epi.append(dict(h=h, model=mod, subset=tag,
                                **metrics(sub.y, sub[mod])))
                m = te.y.notna() & te[mod].notna()
                obs = te.y[m] > thr
                pred = te[mod][m] > thr
                tp = int((obs & pred).sum())
                fp = int((~obs & pred).sum())
                fn = int((obs & ~pred).sum())
                exc.append(dict(
                    h=h, model=mod, thr=thr,
                    pod=tp / (tp + fn) if tp + fn else np.nan,
                    far=fp / (tp + fp) if tp + fp else np.nan,
                    csi=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                    n_events=int(obs.sum())))
    pd.DataFrame(point).to_csv(os.path.join(RES, "point_metrics.csv"),
                               index=False)
    pd.DataFrame(city_rows).to_csv(os.path.join(RES, "point_metrics_city.csv"),
                                   index=False)
    pd.DataFrame(epi).to_csv(os.path.join(RES, "episode_metrics.csv"),
                             index=False)
    pd.DataFrame(exc).to_csv(os.path.join(RES, "exceedance_metrics.csv"),
                             index=False)
    pd.DataFrame(seas).to_csv(os.path.join(RES, "season_metrics.csv"),
                              index=False)
    pd.DataFrame(regi).to_csv(os.path.join(RES, "regime_metrics.csv"),
                              index=False)
    p = pd.DataFrame(point)
    print(p.pivot(index="model", columns="h", values="mae").round(2))
    print()
    print(p.pivot(index="model", columns="h", values="r2").round(3))


if __name__ == "__main__":
    main()
