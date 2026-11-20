# EED-RR: Round Robin scheduling with an elastic time quantum regulated by Jain's fairness index and per-process deficit

Code, data and results for the paper of the same title (Das S, Pranta MMI, Gazi N, Roy A, Islam MS;
submitted to *PLOS ONE*). Everything in the paper's Results section can be regenerated from this repository.

EED-RR enlarges the Round Robin quantum of the dispatched process in proportion to the current unfairness among
active processes (Jain's index of their efficiency ratios) and to that process's efficiency deficit:

```
q_i = TQ_base * (1 + (1 - J) * beta_i * sf),   x_k = cpu_k / (t - a_k),   beta_i = max(0, (mean(x) - x_i) / mean(x))
```

where J and mean(x) are taken over the active processes (arrived, not completed) and TQ_base = (mean + median)/2 of all bursts.

## Quick start

```bash
pip install -r requirements.txt
python tests/test_core.py                     # must print "0 failure(s)"
python run.py --config smoke --out results_smoke   # ~10 s sanity run
```

Reproduce the paper (about 5 minutes on 10 cores, under an hour on a 4-core Kaggle CPU session):

```bash
python run.py --config full --out results_new      # all experiments (uses data/ trace)
python analyze.py --results results_new            # statistics, tables, figures -> results_new/analysis/
python make_paper_tables.py --results results_new --out paper_new   # paper tables, Fig1-4
```

The archived output of exactly these commands is in `results/` and `paper/`. Re-running produces identical
numbers: the study was run independently on Windows (Python 3.14, NumPy 2.5, pandas 3.0) and on Kaggle
(Python 3.13, NumPy 2.1, pandas 2.3), and all 58,890 result rows matched exactly.

On Kaggle: zip this folder, upload it as a Dataset, add it to `notebooks/eedrr_kaggle.ipynb`,
and choose *Save & Run All* (no internet needed).

## Repository layout

| Path | Contents |
|---|---|
| `eedrr_bench/engine.py` | Event-driven single-CPU simulator, metrics, counting conventions (see module docstring) |
| `eedrr_bench/policies.py` | FCFS, SJF, SRTF (oracles); RR family; MARR, EImHLVQTRR, ADRR*, SARR, STS-RR; EED-RR and its ablations; random-boost control |
| `eedrr_bench/workloads.py` | Poisson synthetic workloads, batch (overload), original partitions, SWF trace windows |
| `eedrr_bench/experiments.py` | Experiment design: validation → scaling-factor selection → test; resumable CSV output |
| `eedrr_bench/stats.py` | Paired bootstrap CIs, exact Wilcoxon, Holm correction, d_z, rank-biserial, Hodges–Lehmann |
| `eedrr_bench/analysis.py` | Tables, PLOS-ready TIFF figures with the data behind each, results digest |
| `eedrr_bench/worked_example.py` | The single-dispatch worked example in the paper (`python -m eedrr_bench.worked_example`) |
| `run.py`, `analyze.py`, `make_paper_tables.py` | Entry points (see Quick start) |
| `tests/test_core.py` | Textbook oracles, worked examples, published MARR example, exact reproduction of the earlier version, invariants |
| `data/` | Input data with provenance and licences (see `data/README.md`) |
| `results/` | Archived raw per-instance results, validation selection, statistics and figures (`results/analysis/headline.md` is a readable digest) |
| `paper/` | Generated LaTeX tables, `key_numbers.json`, and Fig1–Fig4 as submitted |
| `notebooks/` | Kaggle notebook that runs tests → experiments → analysis |

## Experimental design (summary)

* **Workloads:** Poisson arrivals at load 0.5/0.7/0.9/0.95 × bursts Gaussian/exponential/lognormal/Pareto/bimodal ×
  n = 100/1000/5000, with 30 independent test seeds per cell; overloaded batch workloads; the original 10-partition
  dataset; 60 windows of the SDSC-SP2 trace; a switch-cost scenario. 2,520 instances in total, 27 policies each.
* **Controls:** RR with the same base quantum and 1.25×/1.5×/2× quantum; equal-budget RR and random boost (same mean
  enlargement as EED-RR); JFI-only and deficit-only ablations; original vs corrected state definitions.
* **Parameter selection:** sf is chosen on separate validation seeds (0–9) by minimising the geometric mean of
  EED-RR/RR mean slowdown, giving **sf = 0.25**. Test seeds are 100–129. sf = 1.5 is reported as a sensitivity result.
* **Statistics:** paired by instance; relative differences with 95% bootstrap CIs; exact Wilcoxon signed-rank with Holm
  correction per metric.

## Key definitions

* **Active set** A(t): processes that have arrived and not completed.
* **Counts:** dispatches, requeues, preemptions (a requeue followed by a different process), process switches, and
  completions, all reported separately. Completions are not counted as context switches.
* **Fairness for every policy:** Jain's index of x at completion (= Jain over inverse slowdowns) and over time.

## Baseline implementation notes

Details and verification are in the paper's S1 Appendix. In brief: MARR sorts each round by remaining burst and
matches the published example of Zohora et al. (PLOS ONE 2024). EImHLVQTRR is reported in two variants
(per round with the half-life completion rule; per dispatch with ceiling rounding), because its source does not
fully specify the update rule. ADRR* is an interpretation, because the source gives no formula. AADRR is not benchmarked (it needs
priorities and agents).

## Relation to the earlier code

An earlier version of this study used the code at <https://github.com/Md-Pranto/Research_Resources>. That code had
errors identified in peer review: Jain's index computed over all n processes, waiting time updated only at dispatch,
the MARR quantum computed once, and completions counted as context switches. This repository replaces it.
`tests/test_core.py::test_reproduces_earlier_version_partition1` reproduces the earlier numbers exactly in a
legacy mode, so the effect of each correction can be inspected.

## Citation

See `CITATION.cff`. If you use the SDSC-SP2 trace, also cite the Parallel Workloads Archive (see `data/README.md`).

## Licence

Code: MIT (see `LICENSE`). Data files keep their own licences (see `data/README.md`).
