"""Re-download and re-extract any yearly raw file that is truncated, then continue."""
import os, subprocess, sys
import duckdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/Vonter/india-cpcb-aqi/releases/download/{y}/cpcb-air-quality-{y}.parquet"
dl = os.path.join(ROOT, "data", "download"); os.makedirs(dl, exist_ok=True)
for y in range(2019, 2025):
    f = os.path.join(ROOT, "data", "raw", "cpcb_%d.parquet" % y)
    try:
        duckdb.connect().execute("SELECT count(*) FROM read_parquet('%s')" % f.replace("\\", "/")).fetchone()
        print(y, "ok", flush=True); continue
    except Exception as e:
        print(y, "invalid, re-downloading:", str(e)[:60], flush=True)
    if os.path.exists(f): os.remove(f)
    tgt = os.path.join(dl, "cpcb-air-quality-%d.parquet" % y)
    if not (os.path.exists(tgt) and os.path.getsize(tgt) > 100_000_000):
        subprocess.run(["curl", "-L", "--retry", "5", "--retry-delay", "10", "-sS", "-o", tgt, URL.format(y=y)], check=True)
    print(y, "download size", os.path.getsize(tgt), flush=True)
    print(y, "downloaded", os.path.getsize(tgt), "bytes", flush=True)
    subprocess.run([sys.executable, "-u", os.path.join(ROOT, "src", "extract_local.py"), str(y)], check=True)
    os.remove(tgt)
print("RAW FILES OK", flush=True)
