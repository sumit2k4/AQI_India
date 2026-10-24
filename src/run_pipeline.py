"""Run the full RGF pipeline sequentially from the repository root.

    python src/run_pipeline.py                 # all stages
    python src/run_pipeline.py --from rgdf.py  # resume at a stage
    python src/run_pipeline.py --only evaluate.py
    python src/run_pipeline.py --revision      # revision analyses (after the main stages)

Each stage's stdout/stderr is written to results/stage_<name>.out and progress to
results/pipeline.log. The run stops at the first failing stage.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
STAGES = ["extract.py", "qc_hourly.py", "train_eval.py", "patch_seasonal_h6.py",
          "train_weighted.py", "rgdf.py", "dl_baselines.py", "transition_analysis.py",
          "evaluate.py", "conformal.py"]
# Analyses added in the revision (need the preds files of the main stages).
REVISION = ["cost_loss.py", "sensitivity.py",
            "dl_baselines_v2.py --models lstm2", "dl_baselines_v2.py --models tcn2",
            "dl_baselines_v2.py --models transformer", "dl_baselines_v2.py --merge",
            "metrics_subset.py", "significance.py"]


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(os.path.join(RES, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, help="first stage to run")
    ap.add_argument("--only", default=None, help="run a single stage")
    ap.add_argument("--revision", action="store_true", help="run the revision analyses")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    allst = REVISION if a.revision else STAGES
    stages = [a.only] if a.only else allst[allst.index(a.start):] if a.start else allst
    for st in stages:
        log("START " + st)
        t0 = time.time()
        script, *args = st.split()
        p = subprocess.run([sys.executable, "-u", os.path.join(SRC, script)] + args, cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        with open(os.path.join(RES, "stage_%s.out" % st.replace(".py", "").replace(" --", "_").replace(" ", "_")), "w",
                  encoding="utf-8") as f:
            f.write((p.stdout or "") + "\n--- STDERR ---\n" + (p.stderr or ""))
        if p.returncode != 0:
            log("FAILED %s rc=%d | %s" % (st, p.returncode, (p.stderr or "")[-800:]))
            sys.exit(1)
        log("DONE %s (%.0f s)" % (st, time.time() - t0))
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
