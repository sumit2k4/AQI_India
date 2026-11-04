"""RGDF: Regime-Gated Decomposition Forecasting for station-level PM2.5.

Mechanism (all causal; fixed a priori, no tuning on cal/test):
  Decomposition   y(t) = B(t) + S(t),  B = 24-h rolling median (>=12 valid)
  Targets         B(t+h), S(t+h)
  Regime          x = pm25(t); fuzzy memberships over three regimes with
                  piecewise-linear transitions 50-70 and 220-280 ug/m3
  Point forecast  yhat = Bhat + sum_r w_r(x) * Shat_r
  Ablations       nogate: single spike model; nodecomp: gated experts on y
  Quantiles       per-regime q05/q95 experts on y, gated identically ->
                  regime-conditional CQR downstream

Appends to results/preds_h{h}.parquet: rgdf, rgdf_nogate, rgdf_nodecomp,
rgdf_q05, rgdf_q95. Writes results/cost_rgdf.csv with fit/predict timings.
"""
import gc
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eval as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

REGIMES = ("low", "mid", "high")


def memberships(x):
    """Fuzzy regime weights; piecewise-linear transitions 50-70, 220-280."""
    x = np.asarray(x, dtype="float64")
    w_low = np.clip((70.0 - x) / 20.0, 0.0, 1.0)
    w_high = np.clip((x - 220.0) / 60.0, 0.0, 1.0)
    w_mid = np.clip(1.0 - w_low - w_high, 0.0, 1.0)
    tot = w_low + w_mid + w_high
    return w_low / tot, w_mid / tot, w_high / tot


def fit_predict(Xtr, ytr, wtr, frames, feat, tag, cost, cat=("sid_code",)):
    t0 = time.time()
    m = lgb.LGBMRegressor(**T.LGB_PARAMS)
    m.fit(Xtr, ytr, sample_weight=wtr, categorical_feature=list(cat))
    fit_s = time.time() - t0
    outs = []
    pred_s = 0.0
    for d in frames:
        t1 = time.time()
        outs.append(m.predict(d[feat]).astype("float32"))
        pred_s += time.time() - t1
    cost.append(dict(model=tag, fit_s=round(fit_s, 1),
                     pred_s=round(pred_s, 1), n_train=len(Xtr)))
    return outs


def fit_predict_q(Xtr, ytr, wtr, frames, feat, q, tag, cost):
    t0 = time.time()
    p = dict(T.LGB_PARAMS)
    p["objective"] = "quantile"
    p["alpha"] = q
    m = lgb.LGBMRegressor(**p)
    m.fit(Xtr, ytr, sample_weight=wtr, categorical_feature=["sid_code"])
    outs = [m.predict(d[feat]).astype("float32") for d in frames]
    cost.append(dict(model=tag, fit_s=round(time.time() - t0, 1),
                     pred_s=0.0, n_train=len(Xtr)))
    return outs


def main():
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    T.meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = T.build_base(df)
    pm = panels["pm25"]
    B = pm.rolling(24, min_periods=12).median().astype("float32")
    S = (pm - B).astype("float32")
    cost = []

    import pyarrow.parquet as pq
    for h in T.HORIZONS:
        fchk = os.path.join(RES, "preds_h%d.parquet" % h)
        if "rgdf" in pq.read_schema(fchk).names:
            print("horizon", h, "already has rgdf, skipping", flush=True)
            continue
        print("=== RGDF horizon", h, "===", flush=True)
        data = T.make_features(panels, full_idx, h)
        yB = B.shift(-h).stack().rename("yB")
        yS = S.shift(-h).stack().rename("yS")
        tgt = pd.concat([yB, yS], axis=1)
        tgt.index.names = ["ts", "sid"]
        data = data.merge(tgt.reset_index(), on=["ts", "sid"], how="left")

        tr = data[(data.ts >= "2019-01-08")
                  & (data.ts <= pd.Timestamp("2023-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        ca = data[(data.ts >= "2024-01-01")
                  & (data.ts <= pd.Timestamp("2024-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        te = data[data.ts >= "2025-01-01"]
        feat = [c for c in data.columns if c not in
                ("y", "yB", "yS", "ts", "sid", "city")]
        frames = (ca, te)
        del data, tgt, yB, yS  # memory: slices are copies
        gc.collect()

        wl_tr, wm_tr, wh_tr = memberships(tr.pm25_lag0)
        Wtr = dict(low=wl_tr, mid=wm_tr, high=wh_tr)
        # severity-aware weighted risk: emphasis on high-concentration targets,
        # ramp anchored at the Severe breakpoint, fixed a priori
        sev_w = np.minimum(1.0 + (tr.y.values / 250.0) ** 2, 20.0)
        Wfr = []
        for d in frames:
            wl, wm, wh = memberships(d.pm25_lag0)
            Wfr.append(dict(low=wl, mid=wm, high=wh))

        # baseline learner (rows with valid yB)
        mB = tr.yB.notna()
        Bhat = fit_predict(tr.loc[mB, feat], tr.loc[mB, "yB"], None,
                           frames, feat, "rgdf_base_h%d" % h, cost)

        # spike experts per regime (membership x severity weights)
        Shat = [np.zeros(len(d), dtype="float64") for d in frames]
        for r in REGIMES:
            m = tr.yS.notna() & (Wtr[r] > 0)
            w = (Wtr[r] * sev_w)[m.values]
            outs = fit_predict(tr.loc[m, feat], tr.loc[m, "yS"], w,
                               frames, feat,
                               "rgdf_spike_%s_h%d" % (r, h), cost)
            for i in range(len(frames)):
                Shat[i] += Wfr[i][r] * outs[i]

        # ablation: no gating (single severity-weighted spike model)
        mS = tr.yS.notna()
        S1 = fit_predict(tr.loc[mS, feat], tr.loc[mS, "yS"],
                         sev_w[mS.values], frames, feat,
                         "rgdf_nogate_spike_h%d" % h, cost)

        # ablation: no decomposition (gated severity-weighted experts on y)
        Ydir = [np.zeros(len(d), dtype="float64") for d in frames]
        for r in REGIMES:
            m = Wtr[r] > 0
            outs = fit_predict(tr.loc[m, feat], tr.loc[m, "y"],
                               (Wtr[r] * sev_w)[m], frames, feat,
                               "rgdf_nodecomp_%s_h%d" % (r, h), cost)
            for i in range(len(frames)):
                Ydir[i] += Wfr[i][r] * outs[i]

        # regime-specific quantile heads on y, gated
        Q = {}
        for q, qtag in ((0.05, "q05"), (0.95, "q95")):
            acc = [np.zeros(len(d), dtype="float64") for d in frames]
            for r in REGIMES:
                m = Wtr[r] > 0
                outs = fit_predict_q(tr.loc[m, feat], tr.loc[m, "y"],
                                     Wtr[r][m], frames, feat, q,
                                     "rgdf_%s_%s_h%d" % (qtag, r, h), cost)
                for i in range(len(frames)):
                    acc[i] += Wfr[i][r] * outs[i]
            Q[qtag] = acc

        f = os.path.join(RES, "preds_h%d.parquet" % h)
        preds = pd.read_parquet(f)
        newcols = ["rgdf", "rgdf_nogate", "rgdf_nodecomp",
                   "rgdf_q05", "rgdf_q95"]
        pieces = []
        for i, d in enumerate(frames):
            p = d[["ts", "sid"]].copy()
            p["rgdf"] = (Bhat[i] + Shat[i]).astype("float32")
            p["rgdf_nogate"] = (Bhat[i] + S1[i]).astype("float32")
            p["rgdf_nodecomp"] = Ydir[i].astype("float32")
            p["rgdf_q05"] = Q["q05"][i].astype("float32")
            p["rgdf_q95"] = Q["q95"][i].astype("float32")
            pieces.append(p)
        newdf = pd.concat(pieces, ignore_index=True)
        preds = preds.drop(columns=[c for c in newcols if c in preds.columns])
        preds = preds.merge(newdf, on=["ts", "sid"], how="left")
        preds.to_parquet(f, index=False)
        pd.DataFrame(cost).to_csv(os.path.join(RES, "cost_rgdf.csv"),
                                  index=False)
        print("appended RGDF columns to", f, flush=True)
        del tr, ca, te
        gc.collect()
    print("RGDF DONE", flush=True)


if __name__ == "__main__":
    main()
