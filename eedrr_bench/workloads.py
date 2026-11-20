"""Workload generators.

* ``synthetic``  : Poisson arrivals at offered load rho, bursts from a named distribution.
                   Every (dist, n, load, seed) is an independent draw.
* ``batch``      : all n processes arrive uniformly within a short window (overload);
                   the regime of the original dataset.
* ``legacy_partitions`` : the ten 1,000-process partitions of the original process_data.csv.
* ``swf_windows``: non-overlapping windows of a Standard Workload Format trace (Parallel
                   Workloads Archive). Job run times are used as CPU bursts and submit times
                   as arrivals, rescaled to a target offered load; a job-level trace is a
                   proxy for CPU bursts.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from .engine import Workload

DISTRIBUTIONS = ("gaussian", "exponential", "lognormal", "bimodal", "pareto")


def sample_bursts(dist: str, n: int, rng: np.random.Generator, mean: float = 100.0) -> np.ndarray:
    if dist == "gaussian":  # mild variability like the original data (CV ~ 0.25)
        b = np.clip(rng.normal(mean, 0.25 * mean, n), 0.05 * mean, 2.0 * mean)
    elif dist == "exponential":  # CV = 1
        b = rng.exponential(mean, n)
    elif dist == "lognormal":  # heavy tail, CV ~ 2.9
        sigma = 1.5
        b = rng.lognormal(np.log(mean) - sigma ** 2 / 2, sigma, n)
    elif dist == "bimodal":  # 80% short interactive, 20% long CPU-bound; mean ~ 0.96*mean
        short = rng.random(n) < 0.8
        b = np.where(short, rng.normal(0.2 * mean, 0.05 * mean, n), rng.normal(4.0 * mean, 0.8 * mean, n))
    elif dist == "pareto":  # Pareto I, alpha = 1.5 (infinite variance), mean = `mean`
        alpha = 1.5
        xm = mean * (alpha - 1) / alpha
        b = xm * (1.0 + rng.pareto(alpha, n))
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    return np.maximum(1.0, np.round(b))


def synthetic(dist: str, n: int, load: float, seed: int, mean: float = 100.0) -> Workload:
    rng = np.random.default_rng([seed, n, int(round(load * 1000)), DISTRIBUTIONS.index(dist)])
    b = sample_bursts(dist, n, rng, mean)
    inter = rng.exponential(b.mean() / load, n)
    a = np.cumsum(inter) - inter[0]
    return Workload(a, b, name=f"{dist}-rho{load:g}-n{n}-s{seed}",
                    meta=dict(family="synthetic", dist=dist, n=n, load=load, seed=seed))


def batch(dist: str, n: int, seed: int, window: float = 100.0, mean: float = 100.0) -> Workload:
    rng = np.random.default_rng([seed, n, 999_999, DISTRIBUTIONS.index(dist)])
    b = sample_bursts(dist, n, rng, mean)
    a = np.sort(rng.uniform(0.0, window, n))
    return Workload(a, b, name=f"batch-{dist}-n{n}-s{seed}",
                    meta=dict(family="batch", dist=dist, n=n, load=np.nan, seed=seed))


def legacy_partitions(csv_path: str | Path, block: int = 1000) -> list[Workload]:
    """The ten contiguous 1,000-process partitions of the original dataset."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    a = np.array([float(r["Arrival Time"]) for r in rows])
    b = np.maximum(1.0, np.array([float(r["Burst time"]) for r in rows]))
    out = []
    for p in range(len(rows) // block):
        sl = slice(p * block, (p + 1) * block)
        out.append(Workload(a[sl], b[sl], name=f"legacy-P{p + 1}",
                            meta=dict(family="legacy", dist="legacy", n=block, load=np.nan, seed=p)))
    return out


def read_swf(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (submit_time, run_time) for jobs with positive run time.

    SWF fields (1-based): 2 = submit time, 4 = run time (-1 = unknown).
    """
    path = Path(path)
    if path.suffix == ".gz":
        with open(path, "rb") as fh:
            magic = fh.read(2)
        if magic != b"\x1f\x8b":
            raise ValueError(f"{path} is not a gzip file (starts with {magic!r}); the download "
                             "probably returned an HTML page. Use the copy in data/ instead.")
    opener = gzip.open if path.suffix == ".gz" else open
    sub, run = [], []
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            f = line.split()
            if len(f) < 4:
                continue
            s, r = float(f[1]), float(f[3])
            if r > 0 and s >= 0:
                sub.append(s)
                run.append(r)
    sub, run = np.asarray(sub), np.asarray(run)
    order = np.argsort(sub, kind="stable")
    return sub[order], run[order]


def swf_windows(path: str | Path, n: int, load: float, max_windows: int | None = None,
                trace_name: str | None = None) -> list[Workload]:
    sub, run = read_swf(path)
    name = trace_name or Path(path).name.split(".swf")[0]
    k = len(sub) // n
    if max_windows is not None:
        k = min(k, max_windows)
    out = []
    for w in range(k):
        s = sub[w * n:(w + 1) * n]
        b = np.maximum(1.0, np.round(run[w * n:(w + 1) * n]))
        a = s - s[0]
        span = a[-1]
        if span <= 0:
            continue
        a = a * (b.sum() / (load * span))  # offered load = sum(b) / arrival span = load
        out.append(Workload(a, b, name=f"{name}-w{w}-rho{load:g}",
                            meta=dict(family="trace", dist=name, n=n, load=load, seed=w)))
    return out
