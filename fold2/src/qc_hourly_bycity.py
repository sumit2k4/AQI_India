"""Bounded-memory, bounded-disk drop-in for qc_hourly.py (identical rules and outputs).
Raw yearly files are split per city with DuckDB (one small scan per city x year, no sort),
then the exact per-station logic of qc_hourly.py runs on one city at a time in pandas."""
import gc
import glob
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc_hourly as Q  # noqa: E402

ROOT, RAW, PROC, RES = Q.ROOT, Q.RAW, Q.PROC, Q.RES
BYC = os.path.join(ROOT, "data", "raw_by_city")


def split_by_city(files):
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'; SET threads=2;")
    cities = set()
    for f in files:
        for r in con.execute("SELECT DISTINCT City FROM read_parquet('%s')" % f.replace("\\", "/")).fetchall():
            if r[0] is not None:
                cities.add(r[0])
    cities = sorted(cities)
    for c in cities:
        d = os.path.join(BYC, c.replace(" ", "_"))
        os.makedirs(d, exist_ok=True)
        for f in files:
            out = os.path.join(d, os.path.basename(f))
            if os.path.exists(out):
                continue
            con.execute("COPY (SELECT * FROM read_parquet('%s') WHERE City = '%s') TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)"
                        % (f.replace("\\", "/"), c.replace("'", "''"), out.replace("\\", "/")))
        print("split city", c, flush=True)
    con.close()
    return cities


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "cpcb_*.parquet")))
    assert files, "run extract.py first"
    cities = split_by_city(files)
    out_rows, meta = [], []
    stuck_total, n_raw = 0, 0
    for c in cities:
        cf = sorted(glob.glob(os.path.join(BYC, c.replace(" ", "_"), "*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in cf], ignore_index=True)
        n_raw += len(df)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        for sid, g in df.groupby("Station ID"):
            # ---- identical to qc_hourly.main() per-station block (fold-2 completeness spans)
            g = g.sort_values("Timestamp").drop_duplicates("Timestamp")
            g = g.set_index("Timestamp")
            pm, nstuck = Q.qc_series(g[Q.PM])
            stuck_total += nstuck
            hourly = pm.resample("1h").agg(["mean", "count"])
            pm25 = hourly["mean"].where(hourly["count"] >= 2)
            rec = pd.DataFrame({"pm25": pm25})
            for src, dst in Q.COVARS.items():
                if src in g.columns:
                    cv = g[src].astype(float)
                    cv[cv < -50] = np.nan
                    rec[dst] = cv.resample("1h").mean()
            rec["sid"] = sid
            rec["city"] = g["City"].iloc[0]
            rec["state"] = g["State"].iloc[0]
            rec["name"] = g["Station Name"].iloc[0]
            out_rows.append(rec.reset_index().rename(columns={"Timestamp": "ts"}))
            tr = pm25["2019":"2022"]
            c23 = pm25["2023":"2023"]
            c24 = pm25["2024":"2024"]

            def comp(x, h):
                return 100.0 * x.notna().sum() / h if h else 0.0

            meta.append({
                "sid": sid, "city": g["City"].iloc[0], "name": g["Station Name"].iloc[0],
                "comp_train": comp(tr, 4 * 8760 + 24),
                "comp_2023": comp(c23, 8760), "comp_2024": comp(c24, 8784),
                "n_stuck_flagged": nstuck,
            })
        print("QC done for", c, "stations so far", len(meta), flush=True)
        del df
        gc.collect()
    meta = pd.DataFrame(meta)
    meta["kept"] = meta.sid.isin(set(pd.read_csv(os.path.join(RES, "station_inclusion_fold1.csv")).query("kept").sid))
    meta.to_csv(os.path.join(RES, "station_inclusion.csv"), index=False)
    allrec = pd.concat(out_rows, ignore_index=True)
    keep_sids = set(meta[meta.kept].sid)
    allrec = allrec[allrec.sid.isin(keep_sids)]
    allrec.to_parquet(os.path.join(PROC, "hourly_pm25.parquet"), index=False)
    pd.DataFrame([{"n_stuck": stuck_total, "pct_stuck": 100.0 * stuck_total / n_raw if n_raw else 0.0,
                   "n_hours": int(allrec.pm25.notna().sum()), "n_raw_15min": n_raw}]).to_csv(
        os.path.join(RES, "qc_stats.csv"), index=False)
    print("stations kept:", len(keep_sids), "/", len(meta))
    print("total 15-min stuck values flagged:", stuck_total)
    print("hourly rows:", len(allrec))
    print(meta.groupby("city").kept.sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
