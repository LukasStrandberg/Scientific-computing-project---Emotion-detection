"""Execute every notebook top-to-bottom in SMOKE_TEST mode, for every MODEL (CPU is fine).

    python tools/run_smoke.py            # all models, reuses existing smoke data / outputs
    python tools/run_smoke.py --fresh    # delete cache/_smoke first (full run, ~15-25 min on a CPU)
    python tools/run_smoke.py --models clustering

Executed copies go to cache/_smoke/executed/<model>/<notebook>.ipynb. MODEL is switched with the
EMO_MODEL environment variable (read by common.config.load_config), SMOKE_TEST with EMO_SMOKE=1.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
PER_MODEL = ["01_preprocess", "02_train", "03_evaluate", "04_patient_videos"]


def execute(name, out_dir, env):
    nb = nbformat.read(ROOT / f"{name}.ipynb", as_version=4)
    old = dict(os.environ)
    os.environ.update(env)
    t0 = time.time()
    try:
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
        status = "ok"
    except Exception as e:  # noqa: BLE001
        status = f"FAILED: {type(e).__name__}: {str(e)[-1500:]}"
    finally:
        os.environ.clear()
        os.environ.update(old)
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out_dir / f"{name}.ipynb")
    print(f"  {name:20s} {status} ({time.time() - t0:.0f} s)", flush=True)
    return status == "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["s2d", "maedfer", "clustering"])
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    smoke = ROOT / "cache" / "_smoke"
    if args.fresh and smoke.exists():
        shutil.rmtree(smoke)
    env = {"EMO_SMOKE": "1", "EMO_DEVICE": os.environ.get("EMO_DEVICE", "cpu")}
    results = [execute("00_setup", smoke / "executed", env)]
    for model in args.models:
        print(f"MODEL = {model}")
        results += [execute(nb, smoke / "executed" / model, {**env, "EMO_MODEL": model}) for nb in PER_MODEL]
    results.append(execute("05_compare", smoke / "executed", env))
    print("ALL OK" if all(results) else f"{results.count(False)} notebook(s) FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
