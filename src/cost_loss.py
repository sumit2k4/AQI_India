"""Relative economic value of exceedance warnings (Richardson 2000), computed
from the contingency tables implied by results/exceedance_metrics.csv and
results/point_metrics.csv (revision, reviewer item 12.5). No predictions
needed, so this runs on the released result tables alone.

For a user with cost-loss ratio a = C/L, base rate s, hit rate H and false
alarm rate F (= FP / (FP + TN)):
    E_forecast = F (1 - s) a + H s a + (1 - H) s
    E_climate  = min(a, s),   E_perfect = s a
    V(a) = (E_climate - E_forecast) / (E_climate - E_perfect)

Output: results/cost_loss.csv  (one row per model x horizon x threshold x a)
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def value(H, F, s, a):
    Ef = F * (1 - s) * a + H * s * a + (1 - H) * s
    Ec = min(a, s)
    Ep = s * a
    return (Ec - Ef) / (Ec - Ep) if Ec > Ep else float("nan")


def main():
    ex = list(csv.DictReader(open(os.path.join(RES, "exceedance_metrics.csv"))))
    pm = {(r["model"], int(r["h"])): r
          for r in csv.DictReader(open(os.path.join(RES, "point_metrics.csv")))}
    out = []
    for r in ex:
        m, h, thr = r["model"], int(r["h"]), r["thr"]
        N = int(float(pm[(m, h)]["n"]))
        ne = float(r["n_events"]); pod = float(r["pod"]); far = float(r["far"])
        TP = pod * ne; FN = ne - TP; FP = TP * far / (1 - far); TN = N - TP - FP - FN
        F = FP / (FP + TN); s = ne / N
        for a in ALPHAS:
            out.append(dict(model=m, h=h, thr=thr, base_rate=round(s, 5),
                            hit_rate=round(pod, 4), false_alarm_rate=round(F, 5),
                            TP=round(TP), FP=round(FP), FN=round(FN), TN=round(TN),
                            cost_loss_ratio=a, value=round(value(pod, F, s, a), 4)))
    with open(os.path.join(RES, "cost_loss.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("saved cost_loss.csv (%d rows)" % len(out))


if __name__ == "__main__":
    main()
