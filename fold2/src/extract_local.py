"""Same extraction as extract.py, but from release files already downloaded to data/download/
(used when DuckDB's HTTP range reads stall). Writes data/raw/cpcb_<year>.parquet."""
import os
import sys
import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import ROOT, RAW, CITIES, WANTED  # noqa: E402

DL = os.path.join(ROOT, "data", "download")


def main():
    os.makedirs(RAW, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='2000MB'; SET threads=4;")
    city_list = ", ".join("'%s'" % c for c in CITIES)
    for y in [int(a) for a in sys.argv[1:]]:
        out = os.path.join(RAW, "cpcb_%d.parquet" % y)
        src = os.path.join(DL, "cpcb-air-quality-%d.parquet" % y).replace("\\", "/")
        if os.path.exists(out):
            print(y, "already extracted, skipping", flush=True)
            continue
        cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet('%s')" % src).fetchall()]
        keep = [c for c in WANTED if c in cols]
        collist = ", ".join('"%s"' % c for c in keep)
        print(y, "columns kept:", len(keep), "->", out, flush=True)
        con.execute("COPY (SELECT %s FROM read_parquet('%s') WHERE City IN (%s)) TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)"
                    % (collist, src, city_list, out.replace("\\", "/")))
        n = con.execute("SELECT count(*) FROM read_parquet('%s')" % out.replace("\\", "/")).fetchone()[0]
        print("  rows:", n, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    main()
