"""Build tables, figures and the results digest from a results directory.

  python analyze.py --results results            # writes results/analysis/
"""
import argparse

from eedrr_bench.analysis import analyze

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    analyze(a.results, a.out, n_boot=a.n_boot)
