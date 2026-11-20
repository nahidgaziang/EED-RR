"""Paired statistics across independent workload replications.

Each replication is one independently generated workload; every policy runs on the same
workload, so comparisons are paired by replication. For proposed x and control y:
  d = x - y (per replication)
  mean_diff, 95% bootstrap CI (percentile, resampling replications)
  rel_pct = 100 * mean(d) / mean(y), with its bootstrap CI
  Wilcoxon signed-rank test (exact for n <= 50 without ties), Holm-adjusted across a family
  d_z = mean(d) / sd(d)                  (paired standardized effect)
  r_rb = rank-biserial correlation       (non-parametric paired effect, -1..1)
  HL   = Hodges-Lehmann estimate         (median of Walsh averages)
  wins = fraction of replications where x < y (lower is better for every metric
         except JFI*, where higher is better; see `lower_is_better`).
No "degrees of freedom" are reported for Wilcoxon.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

HIGHER_IS_BETTER = {"JFI_final", "JFI_t_mean", "JFI_t_min"}


def lower_is_better(metric: str) -> bool:
    return metric not in HIGHER_IS_BETTER


def hodges_lehmann(d: np.ndarray) -> float:
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def paired_summary(x, y, metric: str = "", n_boot: int = 10000, seed: int = 0) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    d = x - y
    out = dict(n_pairs=n, mean_x=x.mean(), mean_y=y.mean(), mean_diff=d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bd = d[idx].mean(axis=1)
    by = y[idx].mean(axis=1)
    out["ci_lo"], out["ci_hi"] = np.percentile(bd, [2.5, 97.5])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_pct"] = 100.0 * d.mean() / y.mean() if y.mean() != 0 else np.nan
        rel = 100.0 * bd / by
    rel = rel[np.isfinite(rel)]
    out["rel_ci_lo"], out["rel_ci_hi"] = (np.percentile(rel, [2.5, 97.5]) if len(rel)
                                          else (np.nan, np.nan))
    sd = d.std(ddof=1) if n > 1 else np.nan
    out["d_z"] = d.mean() / sd if sd and sd > 0 else np.nan
    out["r_rb"] = rank_biserial(d)
    out["HL"] = hodges_lehmann(d)
    better = (d < 0) if lower_is_better(metric) else (d > 0)
    out["wins"] = float(better.mean())
    if n >= 2 and np.any(d != 0):
        method = "exact" if (n <= 50 and len(np.unique(np.abs(d[d != 0]))) == np.sum(d != 0)) else "approx"
        try:
            out["p"] = float(sps.wilcoxon(d, zero_method="wilcox", method=method).pvalue)
        except TypeError:  # older SciPy without `method`
            out["p"] = float(sps.wilcoxon(d, zero_method="wilcox").pvalue)
    else:
        out["p"] = 1.0
    return out


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
