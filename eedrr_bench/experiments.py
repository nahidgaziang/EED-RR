"""Experiment orchestration (validation -> sf selection -> test), resumable.

Stages
  validation : n = val_n, every dist x load, validation seeds; RR + EED-RR over sf_grid.
  select     : choose sf on validation only (pre-specified rule, see `select_sf`).
  test       : synthetic dist x load x n, test seeds (independent of validation seeds).
  batch      : all-arrive-at-once overload scenario.
  legacy     : the ten partitions of the original dataset, plus a legacy-mode reproduction.
  trace      : Standard Workload Format trace windows (if a trace file is supplied).
  switchcost : n = 1000, rho = 0.9, switch cost c = f * TQ_base for f in switch_costs.
  timeseries : JFI(t) series for one seed per dist (rho = 0.9, n = 1000) for figures.
"""
from __future__ import annotations

import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .engine import Engine, default_sample_times
from .policies import (ADRR, EEDRR, FCFS, MARR, RR, SJF, SRTF, EImHLVQTRR, RandomBoostRR,
                       SelfAdjustRR, SmartTimeSliceRR)
from .workloads import DISTRIBUTIONS, batch, legacy_partitions, swf_windows, synthetic

PKG_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CSV = PKG_ROOT / "data" / "process_data.csv"

CONFIGS = {
    "smoke": dict(dists=["exponential", "lognormal", "bimodal"], loads=[0.9], ns=[200],
                  val_n=200, val_seeds=list(range(0, 3)), test_seeds=list(range(100, 104)),
                  sf_grid=[0.5, 1.5, 3.0], batch_n=200, legacy_parts=2,
                  trace_n=500, trace_loads=[0.9], trace_windows=3,
                  switch_costs=[0.01, 0.05]),
    "full": dict(dists=list(DISTRIBUTIONS), loads=[0.5, 0.7, 0.9, 0.95], ns=[100, 1000, 5000],
                 val_n=1000, val_seeds=list(range(0, 10)), test_seeds=list(range(100, 130)),
                 sf_grid=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0], batch_n=1000, legacy_parts=10,
                 trace_n=1000, trace_loads=[0.7, 0.9], trace_windows=30,
                 switch_costs=[0.01, 0.05]),
}

META = ["stage", "split", "family", "dist", "load", "n", "seed", "switch_cost_frac",
        "switch_cost", "idle_mode", "tq_base", "policy", "sf", "wall_s"]
METRICS = ["AWT", "ATT", "W_var", "W_p95", "W_p99", "W_max", "resp_mean", "resp_p95",
           "resp_max", "SD_mean", "SD_p99", "SD_max", "JFI_final", "JFI_t_mean", "JFI_t_min",
           "W_short", "W_medium", "W_long", "SD_short", "SD_long", "resp_short",
           "dispatches", "requeues", "preemptions", "switches", "completions", "makespan",
           "mean_boost", "frac_beta1", "rounds"]
COLUMNS = META + METRICS


# ------------------------------------------------------------------ policy sets
def _named(pol, name):
    pol.name = name
    return pol


def test_policies(sf: float) -> list:
    return [
        FCFS(), SJF(), SRTF(),
        RR(), RR(base="mean", label="RR(mean)"), RR(1.25), RR(1.5), RR(2.0),
        MARR(), EImHLVQTRR(), ADRR(), SelfAdjustRR(), SmartTimeSliceRR(),
        _named(MARR(median="upper"), "MARR(upper-median)"),
        EImHLVQTRR(variant="dispatch", label="EImHLVQTRR(per-dispatch)"),
        EEDRR(sf, label="EED-RR"),
        EEDRR(sf, mode="jfi_only", label="EED[JFI-only]"),
        EEDRR(sf, mode="deficit_only", label="EED[deficit-only]"),
        EEDRR(sf, population="all", waiting="dispatch", label="EED[all,dispatch]"),
        EEDRR(sf, population="all", waiting="continuous", label="EED[all,continuous]"),
        EEDRR(sf, population="active", waiting="dispatch", label="EED[active,dispatch]"),
        EEDRR(1.5, population="all", waiting="dispatch", label="EED-RR-original(sf=1.5)"),
        EEDRR(1.5, label="EED-RR(sf=1.5)"),
    ]


# EED-RR runs whose mean boost defines equal-budget controls: name -> control suffix
MATCHED = {"EED-RR": "", "EED-RR(sf=1.5)": "(sf=1.5)"}


SWITCHCOST_POLICIES = {"FCFS", "SRTF", "RR", "RR(x1.5)", "MARR", "EImHLVQTRR", "EED-RR",
                       "RR-matched", "RandomBoost"}
SERIES_POLICIES = {"FCFS", "SRTF", "RR", "MARR", "EED-RR", "RR-matched"}


# ------------------------------------------------------------------- job specs
def build_workload(spec: dict):
    fam = spec["family"]
    if fam == "synthetic":
        return synthetic(spec["dist"], spec["n"], spec["load"], spec["seed"])
    if fam == "batch":
        return batch(spec["dist"], spec["n"], spec["seed"])
    if fam == "legacy":
        return legacy_partitions(LEGACY_CSV)[spec["seed"]]
    if fam == "trace":
        wins = swf_windows(spec["trace_path"], spec["n"], spec["load"],
                           max_windows=spec["seed"] + 1)
        return wins[spec["seed"]]
    raise ValueError(fam)


def job_key(spec: dict) -> str:
    return "|".join(str(spec.get(k, "")) for k in
                    ("stage", "family", "dist", "load", "n", "seed", "switch_cost_frac", "idle_mode"))


def _row(spec, wl, pol, res, wall):
    r = {k: spec.get(k, "") for k in META}
    r.update(policy=pol.name, sf=getattr(pol, "sf", ""), wall_s=round(wall, 4),
             tq_base=wl.tq_base(), n=wl.n)
    for m in METRICS:
        r[m] = res.get(m, "")
    return r


def run_job(spec: dict) -> dict:
    """Run every policy of a stage on one workload; returns rows (+ optional series)."""
    wl = build_workload(spec)
    S = default_sample_times(wl, k=200 if spec["stage"] == "timeseries" else 40)
    c = spec.get("switch_cost_frac", 0.0) * wl.tq_base()
    spec = dict(spec, switch_cost=c)
    idle = spec.get("idle_mode", "jump")
    rows, series = [], []

    def go(pol, keep=False):
        t0 = time.perf_counter()
        eng = Engine(wl, pol, switch_cost=c, idle_mode=idle, sample_times=S, keep_series=keep)
        res = eng.run()
        rows.append(_row(spec, wl, pol, res, time.perf_counter() - t0))
        if keep:
            series.extend(dict(dist=spec["dist"], load=spec["load"], n=wl.n, seed=spec["seed"],
                               policy=pol.name, t=s, J=J, m=m) for s, J, m in eng.series)
        return res

    stage = spec["stage"]
    if stage == "validation":
        go(RR())
        for sf in spec["sf_grid"]:
            go(EEDRR(sf, label="EED-RR"))
        return dict(rows=rows, series=series)

    if stage == "legacy_repro":  # earlier implementation (clock ticks, all-n, dispatch-only waiting)
        go(RR())
        go(EEDRR(1.5, population="all", waiting="dispatch", label="EED-RR-original(sf=1.5)"))
        return dict(rows=rows, series=series)

    sf = spec["sf"]
    keep_names = SERIES_POLICIES if stage == "timeseries" else set()
    only = SWITCHCOST_POLICIES if stage == "switchcost" else (SERIES_POLICIES if stage == "timeseries" else None)
    boosts = {}
    for pol in test_policies(sf):
        if only is not None and pol.name not in only:
            continue
        res = go(pol, keep=pol.name in keep_names)
        if pol.name in MATCHED:
            boosts[pol.name] = res["mean_boost"]
    for src, suffix in MATCHED.items():
        if src not in boosts:
            continue
        mb = boosts[src]
        for pol in (RR(1.0 + mb, label="RR-matched" + suffix),
                    RandomBoostRR(mb, seed=spec["seed"], label="RandomBoost" + suffix)):
            if only is not None and pol.name not in only:
                continue
            go(pol, keep=pol.name in keep_names)
    return dict(rows=rows, series=series)


# -------------------------------------------------------------------- planning
def plan(stage: str, cfg: dict, sf: float | None, trace_path: str | None) -> list[dict]:
    jobs = []
    if stage == "validation":
        for d in cfg["dists"]:
            for L in cfg["loads"]:
                for s in cfg["val_seeds"]:
                    jobs.append(dict(stage=stage, split="validation", family="synthetic", dist=d,
                                     load=L, n=cfg["val_n"], seed=s, sf_grid=cfg["sf_grid"]))
    elif stage == "test":
        for d in cfg["dists"]:
            for L in cfg["loads"]:
                for n in cfg["ns"]:
                    for s in cfg["test_seeds"]:
                        jobs.append(dict(stage=stage, split="test", family="synthetic", dist=d,
                                         load=L, n=n, seed=s, sf=sf))
    elif stage == "batch":
        for d in cfg["dists"]:
            for s in cfg["test_seeds"]:
                jobs.append(dict(stage=stage, split="test", family="batch", dist=d, load="",
                                 n=cfg["batch_n"], seed=s, sf=sf))
    elif stage == "legacy":
        for p in range(cfg["legacy_parts"]):
            jobs.append(dict(stage=stage, split="legacy", family="legacy", dist="legacy", load="",
                             n=1000, seed=p, sf=sf))
            jobs.append(dict(stage="legacy_repro", split="legacy", family="legacy", dist="legacy",
                             load="", n=1000, seed=p, idle_mode="tick"))
    elif stage == "trace":
        if not trace_path:
            return []
        for L in cfg["trace_loads"]:
            nwin = len(swf_windows(trace_path, cfg["trace_n"], L, max_windows=cfg["trace_windows"]))
            for w in range(nwin):
                jobs.append(dict(stage=stage, split="test", family="trace",
                                 dist=Path(trace_path).name.split(".swf")[0], load=L,
                                 n=cfg["trace_n"], seed=w, sf=sf, trace_path=str(trace_path)))
    elif stage == "switchcost":
        for d in cfg["dists"]:
            for f in cfg["switch_costs"]:
                for s in cfg["test_seeds"]:
                    jobs.append(dict(stage=stage, split="test", family="synthetic", dist=d,
                                     load=0.9, n=1000, seed=s, sf=sf, switch_cost_frac=f))
    elif stage == "timeseries":
        for d in cfg["dists"]:
            jobs.append(dict(stage=stage, split="test", family="synthetic", dist=d, load=0.9,
                             n=1000, seed=cfg["test_seeds"][0], sf=sf))
    else:
        raise ValueError(stage)
    for j in jobs:
        j.setdefault("switch_cost_frac", 0.0)
        j.setdefault("idle_mode", "jump")
    return jobs


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r = dict(r)
            r["switch_cost_frac"] = float(r["switch_cost_frac"] or 0.0)
            for k in ("load",):
                r[k] = float(r[k]) if r[k] not in ("", None) else ""
            r["n"] = int(float(r["n"]))
            r["seed"] = int(float(r["seed"]))
            keys.add(job_key(r))
    return keys


def _norm(spec):
    s = dict(spec)
    s["load"] = float(s["load"]) if s["load"] != "" else ""
    s["switch_cost_frac"] = float(s["switch_cost_frac"])
    return s


def execute(jobs: list[dict], out_csv: Path, workers: int, series_csv: Path | None = None,
            log_every: int = 25) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_csv)
    todo = [j for j in jobs if job_key(_norm(j)) not in done]
    print(f"[{out_csv.name}] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, "
          f"{len(todo)} to run on {workers} worker(s)", flush=True)
    if not todo:
        return
    new_file = not out_csv.exists()
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLUMNS)
    if new_file:
        w.writeheader()
    sfh = sw = None
    if series_csv is not None:
        snew = not series_csv.exists()
        sfh = open(series_csv, "a", newline="")
        sw = csv.DictWriter(sfh, fieldnames=["dist", "load", "n", "seed", "policy", "t", "J", "m"])
        if snew:
            sw.writeheader()
    t0 = time.time()
    try:
        if workers > 1:
            pool = Pool(workers)
            it = pool.imap_unordered(run_job, todo, chunksize=1)
        else:
            pool, it = None, map(run_job, todo)
        for i, out in enumerate(it, 1):
            w.writerows(out["rows"])
            if sw is not None and out["series"]:
                sw.writerows(out["series"])
            if i % log_every == 0 or i == len(todo):
                fh.flush()
                el = time.time() - t0
                print(f"  {i}/{len(todo)} jobs  {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(todo) - i) / 60:.1f} min left", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        fh.close()
        if sfh is not None:
            sfh.close()


# ----------------------------------------------------------------- sf selection
def select_sf(val_csv: Path) -> dict:
    """Pre-specified rule (state it in the paper before looking at test results):
    choose the sf that minimises the geometric mean, over all validation workloads, of
    mean slowdown relative to RR (slowdown weights short jobs, so this trades off
    efficiency against short-job latency and fairness). Ties -> the smaller sf."""
    import pandas as pd

    df = pd.read_csv(val_csv)
    key = ["dist", "load", "n", "seed"]
    rr = df[df.policy == "RR"].set_index(key)
    eed = df[df.policy == "EED-RR"]
    table = []
    for sf, g in eed.groupby("sf"):
        g = g.set_index(key)
        common = g.index.intersection(rr.index)
        row = {"sf": float(sf)}
        for m in ("SD_mean", "AWT", "W_short", "resp_mean", "W_p99", "JFI_final"):
            ratio = g.loc[common, m].astype(float) / rr.loc[common, m].astype(float)
            row[f"gm_ratio_{m}"] = float(np.exp(np.log(ratio).mean()))
        table.append(row)
    table.sort(key=lambda r: (round(r["gm_ratio_SD_mean"], 12), r["sf"]))
    chosen = table[0]["sf"]
    return dict(rule="argmin geometric-mean(SD_mean_EED / SD_mean_RR) over validation workloads",
                chosen_sf=chosen, table=sorted(table, key=lambda r: r["sf"]))


# ------------------------------------------------------------------------ main
def run_all(config: str, out_dir: str | Path, workers: int | None = None,
            stages: list[str] | None = None, trace_path: str | None = None,
            sf_override: float | None = None) -> None:
    cfg = CONFIGS[config]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2))
    stages = stages or ["validation", "test", "batch", "legacy", "trace", "switchcost", "timeseries"]
    (out / "config.json").write_text(json.dumps(dict(config=config, **cfg), indent=2))

    sel_path = out / "sf_selection.json"
    if "validation" in stages:
        execute(plan("validation", cfg, None, None), out / "validation.csv", workers)
    if sf_override is not None:
        sf = sf_override
        sel_path.write_text(json.dumps(dict(rule="override", chosen_sf=sf), indent=2))
    elif sel_path.exists() and "validation" not in stages:
        sf = json.loads(sel_path.read_text())["chosen_sf"]
    else:
        sel = select_sf(out / "validation.csv")
        sel_path.write_text(json.dumps(sel, indent=2))
        sf = sel["chosen_sf"]
    print(f"Scaling factor used for all test stages: sf = {sf}", flush=True)

    for st in stages:
        if st == "validation":
            continue
        jobs = plan(st, cfg, sf, trace_path)
        if not jobs:
            print(f"[{st}] skipped (no jobs; trace file missing?)")
            continue
        execute(jobs, out / f"{st}.csv", workers,
                series_csv=(out / "timeseries_J.csv") if st == "timeseries" else None)
