"""Correctness tests. Run with `pytest -q` or `python tests/test_core.py`."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eedrr_bench import (ADRR, EEDRR, FCFS, MARR, RR, SJF, SRTF, EImHLVQTRR, Policy,  # noqa: E402
                         RandomBoostRR, SelfAdjustRR, SmartTimeSliceRR, Workload, jain, simulate)
from eedrr_bench.engine import Engine, default_sample_times  # noqa: E402
from eedrr_bench.workloads import legacy_partitions, synthetic  # noqa: E402
from eedrr_bench.worked_example import worked_example  # noqa: E402


class FixedQ(Policy):
    def __init__(self, q):
        self.q, self.name = q, f"q={q}"

    def quantum(self, k, eng):
        return self.q


def close(x, y, tol=1e-6):
    return abs(x - y) <= tol * max(1.0, abs(y))


# ---------------------------------------------------------------- textbook oracles
SIL = Workload([0, 0, 0], [24, 3, 3])  # Silberschatz et al., Operating System Concepts


def test_fcfs_textbook():
    assert close(simulate(SIL, FCFS())["AWT"], 17.0)


def test_sjf_textbook():
    assert close(simulate(SIL, SJF())["AWT"], 3.0)


def test_rr_textbook_q4():
    r = simulate(SIL, FixedQ(4))
    assert close(r["AWT"], 17 / 3)
    assert r["dispatches"] == 8 and r["switches"] == 3
    assert r["requeues"] == 5 and r["preemptions"] == 1


def test_srtf_textbook():
    wl = Workload([0, 1, 2, 3], [8, 4, 9, 5])
    r = simulate(wl, SRTF())
    assert close(r["AWT"], 6.5)
    assert r["switches"] == 4 and r["preemptions"] == 1


# ------------------------------------------------------------------- EED-RR checks
def test_eedrr_worked_example():
    """The worked example of the paper (sf = 0.25): J = 0.570, beta_2 = 0.118, TQ = 10.13."""
    r = worked_example()
    J = 0.85 ** 2 / (3 * (0.36 + 0.0625))
    xbar = 0.85 / 3
    beta = (xbar - 0.25) / xbar
    assert close(r["J"], J) and close(r["xbar"], xbar) and close(r["beta"], beta)
    assert close(r["TQ_elastic"], 10 * (1 + (1 - J) * beta * 0.25))
    assert round(r["TQ_elastic"], 2) == 10.13
    assert close(worked_example(sf=1.5)["TQ_elastic"], 10 * (1 + (1 - J) * beta * 1.5))


def test_eedrr_small_trace():
    """Hand-traced: A(0,6), B(0,2), C(1,4); TQ_base = 4, sf = 1.5.
    t=0 A: J=1 -> q=4. t=4 B: x=(1,0,0), J=1/3, beta=1, q=8 -> runs 2.
    t=6 C: x=(2/3,0), J=1/2, beta=1, q=7 -> runs 4. t=10 A alone -> runs 2.
    Completions A=12, B=6, C=10 -> AWT = (6+4+5)/3 = 5."""
    wl = Workload([0, 0, 1], [6, 2, 4])
    r = simulate(wl, EEDRR(sf=1.5))
    assert close(r["AWT"], 5.0)
    assert r["dispatches"] == 4


def test_jain():
    assert close(jain(np.array([1.0, 1.0, 1.0])), 1.0)
    assert close(jain(np.array([1.0, 0.0, 0.0])), 1 / 3)


# --------------------------------------------------- baselines vs published values
def test_marr_matches_published_quanta():
    """Zohora et al. PLOS ONE 2024 (e0304517) Table 4 'random' set; Table 7 lists MARR
    quanta 51, 71, 30, 249 and AWT 522.05, ATT 594.65."""
    bursts = [51, 77, 44, 10, 79, 34, 88, 68, 72, 74, 15, 55, 91, 37, 71, 101, 350, 52, 37, 46]
    wl = Workload(list(range(20)), bursts)
    pol = MARR()
    r = simulate(wl, pol)
    assert pol.quanta_log[:2] == [51, 71], pol.quanta_log
    assert abs(r["AWT"] - 522.05) / 522.05 < 0.005  # standard median: 523.50
    r_up = simulate(wl, MARR(median="upper"))
    assert close(r_up["AWT"], 522.05) and close(r_up["ATT"], 594.65)


def _completions(pol, arrivals, bursts):
    eng = Engine(Workload(arrivals, bursts), pol)
    r = eng.run()
    return r, [float(c) for c in eng.comp]


def test_eimhlvqtrr_matches_published_examples():
    """Abubakar & Isa, UMYU Scientifica 2025 (doi:10.56919/usci.2541.030), both worked examples.
    The per-round rule reproduces them; the per-dispatch variant does not (sensitivity check only)."""
    r = simulate(Workload([0] * 5, [2, 15, 11, 1, 17]), EImHLVQTRR())
    assert close(r["AWT"], 13.4) and close(r["ATT"], 22.6)
    r, comp = _completions(EImHLVQTRR(), [0, 4, 8, 12, 16], [11, 17, 16, 6, 10])
    assert comp == [11, 28, 44, 50, 60] and close(r["AWT"], 18.6)
    assert not close(simulate(Workload([0] * 5, [2, 15, 11, 1, 17]),
                              EImHLVQTRR(variant="dispatch"))["AWT"], 13.4)


def test_sarr_matches_published_examples():
    """Matarneh, Am J Appl Sci 2009 (doi:10.3844/ajassp.2009.1831.1837), cases 1-3 (quanta and AWT).
    Case 4 of the source is internally inconsistent and is not tested."""
    cases = [([0, 0, 0, 0], [20, 40, 60, 80], [50, 25, 25], 62.5),
             ([0, 0, 0, 0], [10, 14, 70, 120], [42, 53, 25], 42.5),
             ([0, 4, 8, 16], [18, 22, 70, 74], [25, 70, 25], 35.0)]
    for a, b, quanta, awt in cases:
        pol = SelfAdjustRR()
        r = simulate(Workload(a, b), pol)
        assert pol.quanta_log == quanta, (b, pol.quanta_log)
        assert close(r["AWT"], awt), (b, r["AWT"])


def test_sts_rr_hand_worked_examples():
    """Hiranwal & Roy 2011 rule (no published example with complete inputs): schedules worked by hand.
    Odd count 9,3,6 -> slice 6 then 3; even count 4,10,7,15 -> slices 9, 3.5, 2.5."""
    r, comp = _completions(SmartTimeSliceRR(), [0, 0, 0], [9, 3, 6])
    assert comp == [18, 3, 9] and close(r["AWT"], 4.0)
    r, comp = _completions(SmartTimeSliceRR(), [0, 0, 0, 0], [4, 10, 7, 15])
    assert comp == [4, 30, 11, 36] and close(r["AWT"], 11.25)


# ---------------------------------------------- reproduction of the earlier version
def test_reproduces_earlier_version_partition1():
    """Legacy mode (clock ticks by 1 when idle, JFI over all n, waiting updated only on
    dispatch) reproduces the earlier version's results for partition P1."""
    p1 = legacy_partitions(ROOT / "data" / "process_data.csv")[0]
    rr = simulate(p1, RR(), idle_mode="tick")
    eed = simulate(p1, EEDRR(sf=1.5, population="all", waiting="dispatch"), idle_mode="tick")
    assert abs(rr["AWT"] - 144036.30) < 0.01 and rr["dispatches"] == 1506
    assert abs(eed["AWT"] - 114014.02) < 0.01 and eed["dispatches"] == 1076


# ------------------------------------------------------------------- invariants
ALL = lambda: [FCFS(), SJF(), SRTF(), RR(), RR(1.5), MARR(), EImHLVQTRR(), ADRR(),  # noqa: E731
               SelfAdjustRR(), SmartTimeSliceRR(), EEDRR(), EEDRR(mode="jfi_only"),
               EEDRR(mode="deficit_only"), EEDRR(population="all", waiting="dispatch"),
               EEDRR(population="all", waiting="continuous"),
               EEDRR(population="active", waiting="dispatch"), RandomBoostRR(0.5, seed=1)]


def test_invariants_and_srtf_optimality():
    for dist in ("exponential", "bimodal", "pareto"):
        wl = synthetic(dist, 150, 0.9, seed=3)
        S = default_sample_times(wl)
        res = {p.name: simulate(wl, p, sample_times=S) for p in ALL()}
        spans = {round(r["makespan"], 6) for r in res.values()}
        assert len(spans) == 1, spans  # work conserving -> identical makespan
        best = res["SRTF"]["AWT"]
        for name, r in res.items():
            assert r["AWT"] >= best - 1e-6, (name, r["AWT"], best)  # SRPT optimal
            assert r["switches"] <= r["dispatches"] and r["preemptions"] <= r["requeues"]
            assert 0 < r["JFI_final"] <= 1 + 1e-12
            assert not math.isnan(r["JFI_t_mean"])


def test_switch_cost_adds_time():
    wl = synthetic("exponential", 100, 0.7, seed=1)
    r0 = simulate(wl, RR())
    r1 = simulate(wl, RR(), switch_cost=1.0)
    assert r1["makespan"] >= r0["makespan"]
    assert r1["AWT"] > r0["AWT"]


def test_engine_rejects_bad_workload():
    try:
        Workload([0, 1], [1, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)
