"""Run the fold-2 pipeline (train 2019-2022 / calibrate 2023 / test 2024) stage by stage."""
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
STAGES = [("extract.py", True), ("qc_hourly_bycity.py", True), ("train_eval.py", True),
          ("patch_seasonal_h6.py", True), ("train_weighted.py", True), ("rgdf_fold2.py", True),
          ("dl_baselines.py", False), ("evaluate.py", True), ("conformal.py", True), ("significance.py", True)]


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(os.path.join(RES, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    os.makedirs(RES, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    stages = STAGES
    if len(sys.argv) > 2 and sys.argv[1] == "--from":
        names = [s for s, _ in STAGES]
        stages = STAGES[names.index(sys.argv[2]):]
        log("RESUME from " + sys.argv[2])
    for st, required in stages:
        log("START " + st)
        t0 = time.time()
        p = subprocess.run([sys.executable, "-u", os.path.join(SRC, st)], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env)
        with open(os.path.join(RES, "stage_%s.out" % st.replace(".py", "")), "w", encoding="utf-8") as f:
            f.write((p.stdout or "") + "\n--- STDERR ---\n" + (p.stderr or ""))
        if p.returncode != 0:
            log("FAILED %s rc=%d | %s" % (st, p.returncode, (p.stderr or "")[-1500:]))
            if required:
                sys.exit(1)
            log("continuing without " + st)
        else:
            log("DONE %s (%.0f s)" % (st, time.time() - t0))
    log("FOLD2 COMPLETE")


if __name__ == "__main__":
    main()
