"""Single-dispatch worked example of EED-RR, as given in the paper.

State at t = 10 with TQ_base = 10 and sf = 1.5; three processes are present:
  P1: a=0, cpu=6  -> x1 = 6/(10-0) = 0.6
  P2: a=2, cpu=2  -> x2 = 2/(10-2) = 0.25   (about to be dispatched)
  P3: a=5, cpu=0  -> x3 = 0/(10-5) = 0
Run ``python -m eedrr_bench.worked_example`` to print the table.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .policies import EEDRR


class _State:
    """Minimal stand-in for Engine exposing what EEDRR.quantum reads."""

    def __init__(self, t, a, cpu, tq_base):
        self.t = t
        self.a = np.asarray(a, float)
        self.cpu = np.asarray(cpu, float)
        self.n = len(a)
        self.act = np.arange(self.n)
        self.nact = self.n
        self.comp = np.full(self.n, np.nan)
        self.tq_base = tq_base
        self.Q = deque()


def worked_example(sf: float = 1.5) -> dict:
    st = _State(t=10.0, a=[0, 2, 5], cpu=[6, 2, 0], tq_base=10.0)
    pol = EEDRR(sf=sf)
    pol.reset(st)
    q = pol.quantum(1, st)  # dispatch P2
    out = dict(pol.last)
    out["x"] = [0.6, 0.25, 0.0]
    out["TQ_elastic"] = q
    out["TQ_applied_if_remaining_5"] = min(q, 5.0)
    return out


if __name__ == "__main__":
    r = worked_example()
    print("x = (0.6, 0.25, 0)")
    print(f"x_bar = {r['xbar']:.5f}   J = {r['J']:.5f}   beta_P2 = {r['beta']:.5f}")
    print(f"boost = (1-J)*beta*sf = {r['boost']:.5f}   TQ = 10 * (1 + boost) = {r['TQ_elastic']:.4f}")
