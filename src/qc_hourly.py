"""QC and hourly aggregation of 15-min CPCB station data.

Rules (documented for the paper's Methods section):
  Q1  Physical range: PM2.5 outside (0, 1500] ug/m3 -> NaN (CPCB instruments
      saturate near 999/1000 at some sites; values >1500 are logger errors).
  Q2  Stuck sensor: runs of >= 8 consecutive identical non-zero 15-min PM2.5
      values (>= 2 h flatline) -> NaN for the whole run.
  Q3  Hourly aggregation: mean of 15-min values; an hour is valid only if
      >= 2 of 4 quarter-hour values survive QC.
  Q4  Station inclusion: >= 60% valid hours in the train span (2019-2023)
      AND >= 60% in 2024 AND >= 60% in 2025.

Output:
  data/processed/hourly_pm25.parquet   (long: sid, city, state, name, ts, pm25 + covariates)
  results/station_inclusion.csv        (per-station completeness + kept flag)
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")

PM = "PM2.5 (µg/m³)"
COVARS = {"PM10 (µg/m³)": "pm10", "NO2 (µg/m³)": "no2", "CO (mg/m³)": "co",
          "Ozone (µg/m³)": "o3", "SO2 (µg/m³)": "so2",
          "AT (°C)": "at", "RH (%)": "rh", "WS (m/s)": "ws", "WD (deg)": "wd",
          "BP (mmHg)": "bp", "SR (W/mt2)": "sr", "RF (mm)": "rf"}


def qc_series(s):
    """Apply Q1+Q2 to one station's 15-min PM2.5 series (time-sorted)."""
    s = s.astype(float)
    s[(s <= 0) | (s > 1500)] = np.nan
    # Q2: flag runs of >=8 identical consecutive non-nan values
    v = s.values
    grp = np.cumsum(np.concatenate([[True], v[1:] != v[:-1]]))
    grp_series = pd.Series(v).groupby(grp)
    runlen = grp_series.transform("size").values
    stuck = (runlen >= 8) & ~np.isnan(v)
    v[stuck] = np.nan
    return pd.Series(v, index=s.index), int(stuck.sum())


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "cpcb_*.parquet")))
    assert files, "run extract.py first"
    frames = []
    stuck_total = 0
    for f in files:
        df = pd.read_parquet(f)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        frames.append(df)
        print("loaded", os.path.basename(f), df.shape)
    df = pd.concat(frames, ignore_index=True)
    del frames

    out_rows = []
    meta = []
    for sid, g in df.groupby("Station ID"):
        g = g.sort_values("Timestamp").drop_duplicates("Timestamp")
        g = g.set_index("Timestamp")
        pm, nstuck = qc_series(g[PM])
        stuck_total += nstuck
        hourly = pm.resample("1H").agg(["mean", "count"])
        pm25 = hourly["mean"].where(hourly["count"] >= 2)
        rec = pd.DataFrame({"pm25": pm25})
        for src, dst in COVARS.items():
            if src in g.columns:
                cv = g[src].astype(float)
                cv[cv < -50] = np.nan
                rec[dst] = cv.resample("1H").mean()
        rec["sid"] = sid
        rec["city"] = g["City"].iloc[0]
        rec["state"] = g["State"].iloc[0]
        rec["name"] = g["Station Name"].iloc[0]
        out_rows.append(rec.reset_index().rename(columns={"Timestamp": "ts"}))

        # completeness per span
        tr = pm25["2019":"2023"]
        c24 = pm25["2024":"2024"]
        c25 = pm25["2025":"2025"]

        def comp(x, h):
            return 100.0 * x.notna().sum() / h if h else 0.0

        meta.append({
            "sid": sid, "city": g["City"].iloc[0], "name": g["Station Name"].iloc[0],
            "comp_train": comp(tr, 5 * 8760 + 24),   # 2019-2023 incl 2020 leap
            "comp_2024": comp(c24, 8784), "comp_2025": comp(c25, 8760),
            "n_stuck_flagged": nstuck,
        })
    meta = pd.DataFrame(meta)
    meta["kept"] = (meta.comp_train >= 60) & (meta.comp_2024 >= 60) & (meta.comp_2025 >= 60)
    meta.to_csv(os.path.join(RES, "station_inclusion.csv"), index=False)

    allrec = pd.concat(out_rows, ignore_index=True)
    keep_sids = set(meta[meta.kept].sid)
    allrec = allrec[allrec.sid.isin(keep_sids)]
    allrec.to_parquet(os.path.join(PROC, "hourly_pm25.parquet"), index=False)
    n_raw = len(df)
    pd.DataFrame([{
        "n_stuck": stuck_total,
        "pct_stuck": 100.0 * stuck_total / n_raw if n_raw else 0.0,
        "n_hours": int(allrec.pm25.notna().sum()),
        "n_raw_15min": n_raw,
    }]).to_csv(os.path.join(RES, "qc_stats.csv"), index=False)
    print("stations kept:", len(keep_sids), "/", len(meta))
    print("total 15-min stuck values flagged:", stuck_total)
    print("hourly rows:", len(allrec))
    print(meta.groupby("city").kept.sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
