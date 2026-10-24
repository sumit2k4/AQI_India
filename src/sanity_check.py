import sys

import duckdb

f = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cpcb_2019.parquet"
con = duckdb.connect()
q = ('SELECT City, count(DISTINCT "Station ID") AS stn, count(*) AS n, '
     'count("PM2.5 (µg/m³)") AS pm25_n '
     "FROM read_parquet('%s') GROUP BY 1 ORDER BY 2 DESC" % f)
print(con.execute(q).df().to_string(index=False))
cols = [r[0] for r in con.execute(
    "DESCRIBE SELECT * FROM read_parquet('%s')" % f).fetchall()]
print("\ncolumns:", cols)
