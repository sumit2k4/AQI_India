"""Train baselines + proposed model for station-level hourly PM2.5 forecasting.

Design (all leakage-free; every feature is measurable at issue time t):
  Horizons: 1, 6, 12, 24 h ahead.
  Splits (blocked, chronological):
      train 2019-2023 | calibration 2024 | test 2025
  Baselines:
      persistence      y(t+h) = pm25(t)
      seasonal_naive   y(t+h) = pm25(t+h-24)  (same hour, previous day)
      climatology      station x month x hour mean over train years
      ridge            linear model, full feature set
      lgbm_uni         LightGBM, own-station lags + calendar only
                       (proxy for standard autoregressive ML in literature)
  Proposed (STPF - SpatioTemporal Pollutant Fusion):
      LightGBM with own-station lags/rolling stats, leave-one-out city-mean
      PM2.5 (spatial signal), co-pollutant and meteorology covariates,
      calendar harmonics, station identity (categorical).
      Plus quantile models (5%, 50%, 95%) for conformalized intervals.

Outputs: results/preds_h{h}.parquet  (split, sid, city, ts, y, model columns)
         results/feature_importance_h{h}.csv
"""
import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

HORIZONS = [1, 6, 12, 24]
LAGS = [0, 1, 2, 3, 6, 12, 23, 24, 48, 168]
ROLLS = [3, 6, 24]
COVARS = ["pm10", "no2", "co", "o3", "so2", "at", "rh", "ws", "bp", "sr"]

LGB_PARAMS = dict(objective="regression_l1", n_estimators=600, learning_rate=0.05,
                  num_leaves=127, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, n_jobs=-1,
                  random_state=42, verbose=-1)


def build_base(df):
    """Wide hourly panel + static per-station features, all float32."""
    df = df.sort_values(["sid", "ts"])
    full_idx = pd.date_range(df.ts.min(), df.ts.max(), freq="1H")
    panels = {}
    for col in ["pm25"] + [c for c in COVARS if c in df.columns]:
        panels[col] = (df.pivot(index="ts", columns="sid", values=col)
                         .reindex(full_idx).astype("float32"))
    return panels, full_idx


def make_features(panels, full_idx, h):
    pm = panels["pm25"]
    sids = pm.columns
    feats = {}
    for lag in LAGS:
        feats["pm25_lag%d" % lag] = pm.shift(lag)
    for w in ROLLS:
        feats["pm25_rmean%d" % w] = pm.rolling(w, min_periods=max(1, w // 2)).mean()
        feats["pm25_rstd%d" % w] = pm.rolling(w, min_periods=max(1, w // 2)).std()
    feats["pm25_d1"] = pm - pm.shift(1)
    feats["pm25_d24"] = pm - pm.shift(24)
    # leave-one-out city mean at time t (spatial signal, no own-station echo)
    city_of = {s: c for s, c in zip(meta_sid.sid, meta_sid.city)}
    cities = pd.Series([city_of[s] for s in sids], index=sids)
    csum = pm.T.groupby(cities).transform("sum").T
    ccnt = pm.notna().T.groupby(cities).transform("sum").T.astype("float32")
    loo = (csum - pm.fillna(0)) / (ccnt - pm.notna().astype("float32")).replace(0, np.nan)
    feats["city_loo_mean"] = loo.astype("float32")
    feats["city_loo_anom"] = (pm - loo).astype("float32")
    for cv in COVARS:
        if cv in panels:
            feats[cv] = panels[cv]
            feats[cv + "_rmean24"] = panels[cv].rolling(24, min_periods=12).mean()

    y = pm.shift(-h)

    # long format
    parts = []
    n_t = len(full_idx)
    hour = full_idx.hour.values
    month = full_idx.month.values
    dow = full_idx.dayofweek.values
    cal = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "mon_sin": np.sin(2 * np.pi * month / 12), "mon_cos": np.cos(2 * np.pi * month / 12),
        "dow": dow.astype("int16"),
    }, index=full_idx).astype("float32")

    for i, s in enumerate(sids):
        d = pd.DataFrame({k: v[s].values for k, v in feats.items()}, index=full_idx)
        d = pd.concat([d, cal], axis=1)
        d["y"] = y[s].values
        d["sid_code"] = np.int16(i)
        d["ts"] = full_idx
        d["sid"] = s
        d = d[d.y.notna() & d.pm25_lag0.notna()]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["city"] = out.sid.map(city_of)
    return out


def main():
    global meta_sid
    os.makedirs(RES, exist_ok=True)
    df = pd.read_parquet(os.path.join(PROC, "hourly_pm25.parquet"))
    meta_sid = df[["sid", "city"]].drop_duplicates()
    panels, full_idx = build_base(df)

    # climatology lookup from train years only: (month, hour, sid) -> mean
    pm_train = panels["pm25"].loc["2019":"2023"]
    clim_tab = pm_train.groupby(
        [pm_train.index.month, pm_train.index.hour]).mean()
    clim_long = clim_tab.stack()
    clim_long.index.names = ["month", "hour", "sid"]

    for h in HORIZONS:
        print("=== horizon", h, "===")
        data = make_features(panels, full_idx, h)
        # trim ends so no split's TARGET (t+h) crosses a split boundary
        tr = data[(data.ts >= "2019-01-08")
                  & (data.ts <= pd.Timestamp("2023-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        ca = data[(data.ts >= "2024-01-01")
                  & (data.ts <= pd.Timestamp("2024-12-31 23:00")
                     - pd.Timedelta(hours=h))]
        te = data[(data.ts >= "2025-01-01")]
        print("rows train/cal/test:", len(tr), len(ca), len(te))

        feat_all = [c for c in data.columns if c not in
                    ("y", "ts", "sid", "city")]
        feat_uni = ([c for c in feat_all if c.startswith("pm25_")]
                    + ["hour_sin", "hour_cos", "mon_sin", "mon_cos", "dow"])
        del data  # memory: tr/ca/te are copies; the full frame is no longer needed
        gc.collect()

        out = {}
        for split, d in (("cal", ca), ("test", te)):
            r = d[["ts", "sid", "city", "y"]].copy()
            # -- persistence / seasonal naive --
            r["persistence"] = d["pm25_lag0"].values
            sn_lag = 24 - h
            r["seasonal_naive"] = d["pm25_lag%d" % sn_lag].values if \
                ("pm25_lag%d" % sn_lag) in d.columns else np.nan
            # -- climatology at target time (vectorized lookup) --
            tgt = d.ts + pd.Timedelta(hours=h)
            lookup_idx = pd.MultiIndex.from_arrays(
                [tgt.dt.month, tgt.dt.hour, d.sid])
            r["climatology"] = clim_long.reindex(lookup_idx).values.astype("float32")
            out[split] = r

        # seasonal naive needs lag(24-h); ensure availability
        Xtr, ytr = tr[feat_all], tr["y"]
        med = Xtr.median(numeric_only=True)

        # ridge
        rg = Ridge(alpha=1.0)
        rg.fit(Xtr.fillna(med), ytr)
        for split, d in (("cal", ca), ("test", te)):
            out[split]["ridge"] = rg.predict(d[feat_all].fillna(med)).astype("float32")

        # lgbm univariate (existing-ML proxy)
        m_uni = lgb.LGBMRegressor(**LGB_PARAMS)
        m_uni.fit(tr[feat_uni], ytr)
        for split, d in (("cal", ca), ("test", te)):
            out[split]["lgbm_uni"] = m_uni.predict(d[feat_uni]).astype("float32")

        # proposed STPF point + quantiles
        m_stpf = lgb.LGBMRegressor(**LGB_PARAMS)
        m_stpf.fit(Xtr, ytr, categorical_feature=["sid_code"])
        imp = pd.DataFrame({"feature": feat_all,
                            "gain": m_stpf.booster_.feature_importance("gain")})
        imp.sort_values("gain", ascending=False).to_csv(
            os.path.join(RES, "feature_importance_h%d.csv" % h), index=False)
        for split, d in (("cal", ca), ("test", te)):
            out[split]["stpf"] = m_stpf.predict(d[feat_all]).astype("float32")

        for q, tag in ((0.05, "q05"), (0.50, "q50"), (0.95, "q95")):
            p = dict(LGB_PARAMS)
            p["objective"] = "quantile"
            p["alpha"] = q
            mq = lgb.LGBMRegressor(**p)
            mq.fit(Xtr, ytr, categorical_feature=["sid_code"])
            for split, d in (("cal", ca), ("test", te)):
                out[split]["stpf_" + tag] = mq.predict(d[feat_all]).astype("float32")

        res = pd.concat([out["cal"].assign(split="cal"),
                         out["test"].assign(split="test")], ignore_index=True)
        res.to_parquet(os.path.join(RES, "preds_h%d.parquet" % h), index=False)
        print("saved preds_h%d.parquet" % h)
        del tr, ca, te, res
        gc.collect()


if __name__ == "__main__":
    main()
