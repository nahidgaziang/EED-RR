"""Run the EED-RR experiments.

Examples
  python run.py --config smoke --out results_smoke   # ~10 s sanity check
  python run.py --config full  --out results_new     # full study (~5 min on 10 cores)

The SDSC-SP2 trace in data/ is used automatically; pass --trace to use another SWF file.
Runs are resumable: re-running with the same --out skips work that is already done.
"""
import argparse
from pathlib import Path

from eedrr_bench.experiments import CONFIGS, run_all

DEFAULT_TRACE = Path(__file__).resolve().parent / "data" / "SDSC-SP2-1998-4.2-cln.swf.gz"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--out", default="results_new")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--trace", default=str(DEFAULT_TRACE) if DEFAULT_TRACE.exists() else None,
                    help="path to an SWF trace (.swf or .swf.gz)")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="subset of: validation test batch legacy trace switchcost timeseries")
    ap.add_argument("--sf", type=float, default=None, help="override the selected scaling factor")
    a = ap.parse_args()
    run_all(a.config, a.out, workers=a.workers, stages=a.stages, trace_path=a.trace,
            sf_override=a.sf)
