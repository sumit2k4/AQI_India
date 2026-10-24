"""Extract candidate-city rows from india-cpcb-aqi yearly release parquets.

Uses DuckDB httpfs so only the needed columns/row-groups are transferred.
Output: data/raw/cpcb_<year>.parquet  (15-min, station-level, candidate cities)

Source: https://github.com/Vonter/india-cpcb-aqi (ODbL). Data (c) CPCB.
"""
import os
import sys
import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
URL = "https://github.com/Vonter/india-cpcb-aqi/releases/download/{y}/cpcb-air-quality-{y}.parquet"

YEARS = list(range(2019, 2026))
CITIES = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad',
          'Ahmedabad', 'Lucknow', 'Patna', 'Jaipur', 'Pune', 'Chandigarh',
          'Gurugram', 'Noida', 'Ghaziabad', 'Varanasi', 'Kanpur', 'Faridabad']

# Columns to keep if present (schema can vary slightly by year)
WANTED = ['Station ID', 'State', 'City', 'Station Name', 'Timestamp',
          'PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'NOx (ppb)',
          'SO2 (µg/m³)', 'CO (mg/m³)', 'Ozone (µg/m³)', 'NH3 (µg/m³)',
          'AT (°C)', 'RH (%)', 'WS (m/s)', 'WD (deg)', 'BP (mmHg)',
          'SR (W/mt2)', 'RF (mm)', 'TOT-RF (mm)']


def main():
    os.makedirs(RAW, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET http_retries=5; SET http_timeout=120000;")
    city_list = ", ".join("'%s'" % c for c in CITIES)

    for y in YEARS:
        out = os.path.join(RAW, "cpcb_%d.parquet" % y)
        if os.path.exists(out):
            print(y, "already extracted, skipping")
            continue
        url = URL.format(y=y)
        cols = [r[0] for r in con.execute(
            "DESCRIBE SELECT * FROM read_parquet('%s')" % url).fetchall()]
        keep = [c for c in WANTED if c in cols]
        collist = ", ".join('"%s"' % c for c in keep)
        print(y, "columns kept:", len(keep), "->", out)
        con.execute("""
            COPY (SELECT %s FROM read_parquet('%s') WHERE City IN (%s))
            TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)
        """ % (collist, url, city_list, out.replace("\\", "/")))
        n = con.execute("SELECT count(*) FROM read_parquet('%s')"
                        % out.replace("\\", "/")).fetchone()[0]
        print("  rows:", n)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    main()
