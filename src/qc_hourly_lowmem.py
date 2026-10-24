"""Low-memory drop-in for qc_hourly.py (identical rules and outputs).

qc_hourly.py concatenates all seven yearly raw files in pandas (about 15 GB
resident), which thrashes on a machine with little free RAM. This version
first partitions the raw files by station with DuckDB (streaming, bounded
memory), then applies exactly the per-station logic of qc_hourly.py to one
station at a time. The original concatenation order (file, row) is restored
before processing so that the first-row metadata and the duplicate-timestamp
handling are the same as in the original script.

Outputs (same as qc_hourly.py):
  data/processed/hourly_pm25.parquet
  results/station_inclusion.csv
  results/qc_stats.csv
"""
import glob
import os
import shutil
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qc_hourly as Q  # noqa: E402  (reuses qc_series, COVARS, PM)

ROOT = Q.ROOT
RAW, PROC, RES = Q.RAW, Q.PROC, Q.RES
PART = os.path.join(ROOT, "data", "raw_by_station")
TMP = os.path.join(ROOT, "data", "duck_tmp")


def partition_by_station(files):
    if os.path.isdir(PART) and glob.glob(os.path.join(PART, "*", "*.parquet")):
        print("partitions already exist, skipping", flush=True)
        return
    shutil.rmtree(PART, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'; SET threads=4;")
    con.execute("SET temp_directory='%s';" % TMP.replace("\\", "/"))
    con.execute("SET preserve_insertion_order=false;")
    flist = ", ".join("'%s'" % f.replace("\\", "/") for f in files)
    con.execute("""
        COPY (SELECT *, filename AS _file, file_row_number AS _row
              FROM read_parquet([%s], filename=true, file_row_number=true))
        TO '%s' (FORMAT PARQUET, PARTITION_BY ("Station ID"), COMPRESSION ZSTD)
    """ % (flist, PART.replace("\\", "/")))
    con.close()
    shutil.rmtree(TMP, ignore_errors=True)
    print("partitioned into", len(os.listdir(PART)), "station folders", flush=True)


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "cpcb_*.parquet")))
    assert files, "run extract.py first"
    partition_by_station(files)

    folders = {}
    n_raw = 0
    for d in os.listdir(PART):
        key = d.split("=", 1)[1] if "=" in d else d
        n = duckdb.sql("SELECT count(*) FROM read_parquet('%s')"
                       % os.path.join(PART, d, "*.parquet").replace("\\", "/")).fetchone()[0]
        n_raw += n
        if key == "__HIVE_DEFAULT_PARTITION__":
            continue  # NULL station id: dropped by groupby in the original
        folders[key] = os.path.join(PART, d)
    print("raw 15-min rows:", n_raw, "stations:", len(folders), flush=True)

    out_parts, meta = [], []
    stuck_total = 0
    for i, sid in enumerate(sorted(folders)):
        g = pd.read_parquet(folders[sid])
        g["Station ID"] = sid
        g = g.sort_values(["_file", "_row"], kind="stable").drop(columns=["_file", "_row"])
        g = g.reset_index(drop=True)
        g["Timestamp"] = (pd.to_datetime(g["Timestamp"], utc=True)
                          .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
        # ---- identical to qc_hourly.main() from here on
        g = g.sort_values("Timestamp").drop_duplicates("Timestamp")
        g = g.set_index("Timestamp")
        pm, nstuck = Q.qc_series(g[Q.PM])
        stuck_total += nstuck
        hourly = pm.resample("1H").agg(["mean", "count"])
        pm25 = hourly["mean"].where(hourly["count"] >= 2)
        rec = pd.DataFrame({"pm25": pm25})
        for src, dst in Q.COVARS.items():
            if src in g.columns:
                cv = g[src].astype(float)
                cv[cv < -50] = np.nan
                rec[dst] = cv.resample("1H").mean()
        rec["sid"] = sid
        rec["city"] = g["City"].iloc[0]
        rec["state"] = g["State"].iloc[0]
        rec["name"] = g["Station Name"].iloc[0]
        out_parts.append(rec.reset_index().rename(columns={"Timestamp": "ts"}))

        tr = pm25["2019":"2023"]
        c24 = pm25["2024":"2024"]
        c25 = pm25["2025":"2025"]

        def comp(x, h):
            return 100.0 * x.notna().sum() / h if h else 0.0

        meta.append({
            "sid": sid, "city": g["City"].iloc[0], "name": g["Station Name"].iloc[0],
            "comp_train": comp(tr, 5 * 8760 + 24),
            "comp_2024": comp(c24, 8784), "comp_2025": comp(c25, 8760),
            "n_stuck_flagged": nstuck,
        })
        if (i + 1) % 20 == 0:
            print("  processed %d stations" % (i + 1), flush=True)
        del g
    meta = pd.DataFrame(meta)
    meta["kept"] = (meta.comp_train >= 60) & (meta.comp_2024 >= 60) & (meta.comp_2025 >= 60)
    meta.to_csv(os.path.join(RES, "station_inclusion.csv"), index=False)

    allrec = pd.concat(out_parts, ignore_index=True)
    keep_sids = set(meta[meta.kept].sid)
    allrec = allrec[allrec.sid.isin(keep_sids)]
    allrec.to_parquet(os.path.join(PROC, "hourly_pm25.parquet"), index=False)
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
