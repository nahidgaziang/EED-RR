"""Event-driven single-CPU scheduling simulator shared by every policy.

All schedulers (FCFS, SJF, SRTF, every Round Robin variant and every EED-RR variant)
run on this one engine, so they differ only in *which* ready process is dispatched
(``Policy.select``) and *for how long* (``Policy.quantum``).

Conventions (stated once, used everywhere):
* Processes become ready at their arrival time. Arrivals that occur while a process
  runs are enqueued *before* that process is re-queued (the usual RR convention).
* When the ready queue is empty the clock jumps to the next arrival
  (``idle_mode="jump"``). ``idle_mode="tick"`` reproduces the original EED-RR code,
  which advanced the clock one unit at a time.
* ``dispatches``  = number of times a process was given the CPU.
  ``completions`` = n.
  ``requeues``    = times a process went back to the queue unfinished
                   (counted as preemptions in the earlier version).
  ``preemptions`` = requeues after which a *different* process was dispatched.
  ``switches``    = dispatches of a process different from the previously running one
                   (the first dispatch is not counted).
* ``switch_cost`` (time units) is charged on every switch; the CPU does no useful work
  during it.
* Fairness over time: at fixed sample times the engine computes Jain's index of
  x_j(s) = cpu_j(s) / (s - a_j) over every process that has arrived and not yet
  completed. This is the same quantity for every algorithm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


@dataclass
class Workload:
    """A set of processes (arrival time, CPU burst). Sorted by arrival on creation."""

    arrival: np.ndarray
    burst: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        a = np.asarray(self.arrival, dtype=float)
        b = np.asarray(self.burst, dtype=float)
        if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
            raise ValueError("arrival and burst must be equal-length 1-D arrays")
        if np.any(b <= 0):
            raise ValueError("bursts must be positive")
        order = np.argsort(a, kind="stable")
        self.arrival = a[order]
        self.burst = b[order]

    @property
    def n(self) -> int:
        return len(self.arrival)

    def tq_base(self) -> float:
        """Base quantum used by RR and EED-RR: (mean + median) / 2 of all bursts.

        This requires every burst length in advance (offline knowledge).
        """
        return (float(np.mean(self.burst)) + float(np.median(self.burst))) / 2.0

    def work_conserving_makespan(self) -> float:
        t = 0.0
        for a, b in zip(self.arrival, self.burst):
            t = max(t, a) + b
        return t


def jain(x: np.ndarray) -> float:
    """Jain's fairness index; 1.0 for an empty or all-zero vector."""
    if len(x) == 0:
        return 1.0
    s2 = float(np.dot(x, x))
    if s2 <= 0:
        return 1.0
    s1 = float(np.sum(x))
    return s1 * s1 / (len(x) * s2)


class Policy:
    """Base class. Subclasses override ``quantum`` and optionally the hooks."""

    name = "policy"

    def reset(self, eng: "Engine") -> None:  # called once before the run
        pass

    def select(self, eng: "Engine") -> int:  # remove and return the process to run
        return eng.Q.popleft()

    def quantum(self, k: int, eng: "Engine") -> float:
        raise NotImplementedError

    def after_run(self, k: int, ran: float, eng: "Engine") -> None:
        pass

    def info(self) -> dict:  # extra per-run diagnostics (e.g. mean boost)
        return {}


class Engine:
    def __init__(self, wl: Workload, policy: Policy, switch_cost: float = 0.0,
                 idle_mode: str = "jump", sample_times: np.ndarray | None = None,
                 keep_series: bool = False):
        if idle_mode not in ("jump", "tick"):
            raise ValueError(idle_mode)
        self.wl = wl
        self.policy = policy
        self.switch_cost = float(switch_cost)
        self.idle_mode = idle_mode
        self.a = wl.arrival
        self.b = wl.burst
        self.n = wl.n
        self.tq_base = wl.tq_base()
        self.samples = np.asarray(sample_times if sample_times is not None else [], float)
        self.keep_series = keep_series

    # ------------------------------------------------------------------ helpers
    def _admit(self, t: float) -> None:
        a, Q = self.a, self.Q
        while self.i < self.n and a[self.i] <= t:
            j = self.i
            Q.append(j)
            self.act[self.nact] = j
            self.pos[j] = self.nact
            self.nact += 1
            self.i += 1

    def _deactivate(self, k: int) -> None:
        p = self.pos[k]
        last = self.act[self.nact - 1]
        self.act[p] = last
        self.pos[last] = p
        self.pos[k] = -1
        self.nact -= 1

    def _sample(self, s: float, k: int, run_start: float) -> None:
        """Jain's index of x_j(s) over processes present at time s."""
        a = self.a
        idx = self.act[: self.nact]
        age = s - a[idx]
        c = self.cpu[idx].copy()
        if k >= 0 and s > run_start:
            c[self.pos[k]] += s - run_start
        ok = age > EPS
        x = c[ok] / age[ok]
        j_end = int(np.searchsorted(a, s, side="left"))  # pending arrivals with a < s
        n_pend = max(0, j_end - self.i)
        if n_pend:
            x = np.concatenate([x, np.zeros(n_pend)])
        if len(x) == 0:
            return
        J = jain(x)
        self.J_samples.append(J)
        if self.keep_series:
            self.series.append((s, J, len(x)))

    def _sample_through(self, t_end: float, k: int, run_start: float) -> None:
        S = self.samples
        while self.sp < len(S) and S[self.sp] <= t_end:
            self._sample(S[self.sp], k, run_start)
            self.sp += 1

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        n, a, b = self.n, self.a, self.b
        self.rem = b.copy()
        self.cpu = np.zeros(n)
        self.comp = np.full(n, np.nan)
        self.first = np.full(n, np.nan)
        self.act = np.empty(n, dtype=np.int64)
        self.pos = np.full(n, -1, dtype=np.int64)
        self.nact = 0
        self.Q = deque()
        self.i = 0
        self.t = 0.0
        self.sp = 0
        self.J_samples: list[float] = []
        self.series: list[tuple] = []
        dispatches = requeues = preemptions = switches = 0
        last = -1
        pending_preempt = -1
        self.policy.reset(self)

        t = 0.0
        while self.i < n or self.Q:
            self._admit(t)
            if not self.Q:
                nxt = a[self.i]
                self._sample_through(nxt - EPS, -1, nxt)  # idle period: nothing active
                t = t + 1.0 if self.idle_mode == "tick" else nxt
                continue
            self.t = t
            k = self.policy.select(self)
            if pending_preempt >= 0:
                if k != pending_preempt:
                    preemptions += 1
                pending_preempt = -1
            if last >= 0 and k != last:
                switches += 1
                if self.switch_cost > 0:
                    self._sample_through(t + self.switch_cost, -1, np.inf)
                    t += self.switch_cost
                    self.t = t
            if np.isnan(self.first[k]):
                self.first[k] = t - a[k]
            q = float(self.policy.quantum(k, self))
            run = min(q, self.rem[k])
            if not run > 0:
                raise RuntimeError(f"{self.policy.name}: non-positive run time {run}")
            self._sample_through(t + run, k, t)
            t += run
            self.t = t
            self.rem[k] -= run
            self.cpu[k] += run
            dispatches += 1
            last = k
            self.policy.after_run(k, run, self)
            self._admit(t)
            if self.rem[k] > EPS:
                self.Q.append(k)
                requeues += 1
                pending_preempt = k
            else:
                self.rem[k] = 0.0
                self.comp[k] = t
                self._deactivate(k)

        counters = dict(dispatches=dispatches, requeues=requeues,
                        preemptions=preemptions, switches=switches, completions=n)
        return summarize(self.wl, self.comp, self.first, counters,
                         np.asarray(self.J_samples), self.policy.info())


def summarize(wl: Workload, comp: np.ndarray, first: np.ndarray, counters: dict,
              J_samples: np.ndarray, extra: dict) -> dict:
    a, b = wl.arrival, wl.burst
    if np.isnan(comp).any():
        raise RuntimeError("some processes did not complete")
    T = comp - a
    W = T - b
    W = np.where(np.abs(W) < 1e-7, 0.0, W)
    if (W < -1e-6).any():
        raise RuntimeError("negative waiting time")
    SD = T / b
    q1, q2 = np.quantile(b, [1 / 3, 2 / 3])
    short, long_ = b <= q1, b > q2
    med = ~short & ~long_
    out = dict(
        n=wl.n,
        makespan=float(comp.max()),
        AWT=float(W.mean()),
        ATT=float(T.mean()),
        W_var=float(W.var()),
        W_p95=float(np.quantile(W, 0.95)),
        W_p99=float(np.quantile(W, 0.99)),
        W_max=float(W.max()),
        resp_mean=float(first.mean()),
        resp_p95=float(np.quantile(first, 0.95)),
        resp_max=float(first.max()),
        SD_mean=float(SD.mean()),
        SD_p99=float(np.quantile(SD, 0.99)),
        SD_max=float(SD.max()),
        JFI_final=jain(b / T),  # = Jain over x_i at completion (x_i = burst/turnaround)
        JFI_t_mean=float(J_samples.mean()) if len(J_samples) else np.nan,
        JFI_t_min=float(J_samples.min()) if len(J_samples) else np.nan,
        W_short=float(W[short].mean()) if short.any() else np.nan,
        W_medium=float(W[med].mean()) if med.any() else np.nan,
        W_long=float(W[long_].mean()) if long_.any() else np.nan,
        SD_short=float(SD[short].mean()) if short.any() else np.nan,
        SD_long=float(SD[long_].mean()) if long_.any() else np.nan,
        resp_short=float(first[short].mean()) if short.any() else np.nan,
    )
    out.update(counters)
    out.update(extra)
    return out


def simulate(wl: Workload, policy: Policy, **kw) -> dict:
    return Engine(wl, policy, **kw).run()


def default_sample_times(wl: Workload, k: int = 40) -> np.ndarray:
    """k evenly spaced instants strictly inside [first arrival, makespan]."""
    end = wl.work_conserving_makespan()
    return np.linspace(wl.arrival[0], end, k + 2)[1:-1]
