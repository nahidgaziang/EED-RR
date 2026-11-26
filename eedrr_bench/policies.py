"""Scheduling policies. Each class documents the source definition it implements and
any implementation choice the source leaves open."""
from __future__ import annotations

import math

import numpy as np

from .engine import EPS, Policy


def _round_half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


# --------------------------------------------------------------------------- oracles
class FCFS(Policy):
    name = "FCFS"

    def quantum(self, k, eng):
        return math.inf


class SJF(Policy):
    """Non-preemptive shortest-job-first. Oracle: needs exact burst lengths."""

    name = "SJF"

    def select(self, eng):
        b = eng.b
        k = min(eng.Q, key=lambda j: (b[j], j))
        eng.Q.remove(k)
        return k

    def quantum(self, k, eng):
        return math.inf


class SRTF(Policy):
    """Preemptive shortest-remaining-time-first. Oracle: needs exact burst lengths.

    A running process is re-evaluated at every arrival.
    """

    name = "SRTF"

    def select(self, eng):
        rem = eng.rem
        k = min(eng.Q, key=lambda j: (rem[j], j))
        eng.Q.remove(k)
        return k

    def quantum(self, k, eng):
        nxt = eng.a[eng.i] if eng.i < eng.n else math.inf
        return max(nxt - eng.t, EPS) if nxt < math.inf else math.inf


# ------------------------------------------------------------------------- RR family
class RR(Policy):
    """Round Robin with a fixed quantum q = mult x base.

    base="mm"   -> (mean + median)/2 of all bursts, the same base as EED-RR.
    base="mean" -> mean of all bursts.
    """

    def __init__(self, mult: float = 1.0, base: str = "mm", label: str | None = None):
        self.mult, self.base = float(mult), base
        self.name = label or (f"RR(x{mult:g})" if mult != 1 else "RR")

    def reset(self, eng):
        base = eng.tq_base if self.base == "mm" else float(np.mean(eng.b))
        self.q = self.mult * base

    def quantum(self, k, eng):
        return self.q


class _RoundPolicy(Policy):
    """Helper for algorithms that fix one quantum per *round*.

    A round starts when the previous round's processes have all been dispatched once;
    it covers exactly the processes in the ready queue at that moment. Processes that
    arrive during a round join the tail and are served in the next round.
    """

    def reset(self, eng):
        self.left = 0
        self.tq = None
        self.quanta_log = []

    def _maybe_new_round(self, k, eng):
        if self.left == 0:
            members = [k] + list(eng.Q)
            self.tq = self.round_quantum([eng.rem[j] for j in members], eng)
            self.left = len(members)
            self.quanta_log.append(self.tq)
        self.left -= 1

    def info(self):
        return dict(rounds=len(self.quanta_log))

    def round_quantum(self, remaining: list[float], eng) -> float:
        raise NotImplementedError


class MARR(_RoundPolicy):
    """Median-Average Round Robin (MARR).

    Source: Sakshi et al., Alexandria Eng J 2022;61(12):10527-38,
    doi:10.1016/j.aej.2022.04.006. Per round: the ready queue is sorted by remaining
    burst (ascending) and q = round((median + mean)/2) of the remaining bursts of the
    processes in that round.
    On the 20-process example of Zohora et al. (PLOS ONE 2024, e0304517, Tables 4 and 7)
    this reproduces the published quanta 51 and 71 and, with median="upper", the published
    AWT 522.05 and ATT 594.65; the standard median gives AWT 523.50.
    """

    name = "MARR"

    def __init__(self, median: str = "standard", sort: bool = True):
        assert median in ("standard", "upper")
        self.median, self.sort = median, sort

    def select(self, eng):
        if self.sort and self.left == 0 and len(eng.Q) > 1:
            rem = eng.rem
            ordered = sorted(eng.Q, key=lambda j: (rem[j], j))
            eng.Q.clear()
            eng.Q.extend(ordered)
        return eng.Q.popleft()

    def round_quantum(self, remaining, eng):
        if self.median == "standard":
            med = float(np.median(remaining))
        else:
            med = float(sorted(remaining)[len(remaining) // 2])
        v = (med + float(np.mean(remaining))) / 2.0
        return max(1, _round_half_up(v))

    def quantum(self, k, eng):
        self._maybe_new_round(k, eng)
        return self.tq


class EImHLVQTRR(_RoundPolicy):
    """Enhanced and Improved Half-Life Variable Quantum Time Round Robin.

    Source: Abubakar SE, Isa I. UMYU Scientifica 2025;4(1):305-16,
    doi:10.56919/usci.2541.030. TQ = sqrt(mean(RBT) * min(RBT)) over the ready queue.
    The per-round variant reproduces both worked examples of the source exactly (see
    tests/test_core.py); the per-dispatch variant does not and is kept as a sensitivity check:

    variant="round":    TQ = max(1, floor(.)) once per round; a process whose remaining
                        burst is <= 2*TQ runs to completion (the half-life rule of the
                        HLVQTRR/ImHLVQTRR family, Sci World J 2022;17(2):248-53).
    variant="dispatch": TQ = max(1, ceil(.)) recomputed at every dispatch over the ready
                        queue including the dispatched process; no completion rule.
    """

    name = "EImHLVQTRR"

    def __init__(self, completion_factor: float = 2.0, variant: str = "round",
                 label: str | None = None):
        assert variant in ("round", "dispatch")
        self.cf, self.variant = completion_factor, variant
        if label:
            self.name = label

    def round_quantum(self, remaining, eng):
        return max(1, int(math.floor(math.sqrt(np.mean(remaining) * min(remaining)))))

    def quantum(self, k, eng):
        if self.variant == "dispatch":
            rem = [eng.rem[k]] + [eng.rem[j] for j in eng.Q]
            return max(1, math.ceil(math.sqrt(np.mean(rem) * min(rem))))
        self._maybe_new_round(k, eng)
        r = eng.rem[k]
        return r if r <= self.cf * self.tq else self.tq


class ADRR(Policy):
    """Absolute Difference Based Time Quantum Round Robin (interpretation, ADRR*).

    Source: Borate et al., IJARSCT 2025;5(3), doi:10.48175/IJARSCT-29824. The source
    describes quanta "based on the absolute difference between consecutive burst times"
    with decision-tree burst prediction, but gives no formula. Interpretation used here:
    predicted burst = 1.1 x burst; base = mean(predicted); q_i = base + |p_(r) - p_(r-1)|,
    where r is the rank of process i's predicted burst.
    """

    name = "ADRR*"

    def reset(self, eng):
        pred = eng.b * 1.1
        order = np.argsort(pred, kind="stable")
        sp = pred[order]
        diff = np.concatenate([[0.0], np.abs(np.diff(sp))])
        # equal predictions share the rank of their first occurrence
        first_rank = {}
        for r, v in enumerate(sp):
            first_rank.setdefault(v, r)
        self.q = np.array([sp.mean() + diff[first_rank[p]] for p in pred])

    def quantum(self, k, eng):
        return self.q[k]


class SelfAdjustRR(_RoundPolicy):
    """Self-Adjustment Round Robin (SARR).

    Source: Matarneh R, Am J Appl Sci 2009;6(10):1831-7. Per round, q = median of the
    remaining bursts in the ready queue, with a lower bound floor_q (25 in the source).
    """

    name = "SARR"

    def __init__(self, floor_q: float = 25.0):
        self.floor_q = floor_q

    def round_quantum(self, remaining, eng):
        return max(self.floor_q, float(np.median(remaining)))

    def quantum(self, k, eng):
        self._maybe_new_round(k, eng)
        return self.tq


class SmartTimeSliceRR(_RoundPolicy):
    """Adaptive Round Robin with shortest-burst ordering and a smart time slice (STS-RR).

    Source: Hiranwal S, Roy KC, Int J Comput Sci Commun 2011;2(2):319-23. Per round the
    ready queue is sorted by remaining burst (ascending); the time slice is the middle
    burst if the queue length is odd, otherwise the mean burst.
    """

    name = "STS-RR"

    def select(self, eng):
        if self.left == 0 and len(eng.Q) > 1:
            rem = eng.rem
            ordered = sorted(eng.Q, key=lambda j: (rem[j], j))
            eng.Q.clear()
            eng.Q.extend(ordered)
        return eng.Q.popleft()

    def round_quantum(self, remaining, eng):
        r = sorted(remaining)
        return r[len(r) // 2] if len(r) % 2 == 1 else float(np.mean(r))

    def quantum(self, k, eng):
        self._maybe_new_round(k, eng)
        return self.tq


# --------------------------------------------------------------------- EED-RR family
class EEDRR(Policy):
    """EED-RR and its ablations.

    q_i = TQ_base * (1 + boost_i), with
      mode="full"          boost = (1 - J) * beta_i * sf      (EED-RR)
      mode="jfi_only"      boost = (1 - J) * sf               (uniform boost, no deficit)
      mode="deficit_only"  boost = beta_i * sf                 (no global JFI signal)
    where x_j is the efficiency ratio, J is Jain's index of x over the population, and
    beta_i = max(0, (mean(x) - x_i) / mean(x)).

    population: "active" = arrived and not completed (default);
                "all"    = all n processes; processes not yet arrived count as 0.5 and
                           completed ones keep their last value.
    waiting:    "continuous" = x_j = cpu_j / (t - a_j) (default);
                "dispatch"   = waiting is accumulated only when j is dispatched;
                               x = 0.5 when cpu = wait = 0.
    The earlier version of EED-RR corresponds to EEDRR(sf=1.5, population="all",
    waiting="dispatch").
    """

    def __init__(self, sf: float = 1.5, mode: str = "full", population: str = "active",
                 waiting: str = "continuous", label: str | None = None):
        assert mode in ("full", "jfi_only", "deficit_only")
        assert population in ("active", "all")
        assert waiting in ("continuous", "dispatch")
        self.sf, self.mode, self.population, self.waiting = float(sf), mode, population, waiting
        if label:
            self.name = label
        else:
            tag = {"full": "EED-RR", "jfi_only": "EED[JFI-only]",
                   "deficit_only": "EED[deficit-only]"}[mode]
            legacy = "" if (population, waiting) == ("active", "continuous") else \
                f"[{population},{waiting}]"
            self.name = f"{tag}{legacy}(sf={sf:g})"

    def reset(self, eng):
        n = eng.n
        self.boost_sum = 0.0
        self.boost_n = 0
        self.beta_one = 0
        if self.waiting == "dispatch":
            self.accw = np.zeros(n)
            self.last_end = eng.a.copy()

    def _x(self, eng, t):
        a, cpu = eng.a, eng.cpu
        if self.population == "active":
            idx = eng.act[: eng.nact]
            if self.waiting == "continuous":
                age = t - a[idx]
                c = cpu[idx]
                x = np.where(age > EPS, c / np.where(age > EPS, age, 1.0), 0.5)
            else:
                c, w = cpu[idx], self.accw[idx]
                s = c + w
                x = np.where(s > 0, c / np.where(s > 0, s, 1.0), 0.5)
        else:
            if self.waiting == "continuous":
                done = ~np.isnan(eng.comp)
                end = np.where(done, eng.comp, t)
                age = end - a
                x = np.where(age > EPS, cpu / np.where(age > EPS, age, 1.0), 0.5)
                x = np.where(a > t, 0.5, x)
            else:
                s = cpu + self.accw
                x = np.where(s > 0, cpu / np.where(s > 0, s, 1.0), 0.5)
        return x

    def _xk(self, k, eng, t):
        if self.waiting == "continuous":
            age = t - eng.a[k]
            return eng.cpu[k] / age if age > EPS else 0.5
        s = eng.cpu[k] + self.accw[k]
        return eng.cpu[k] / s if s > 0 else 0.5

    def quantum(self, k, eng):
        t = eng.t
        if self.waiting == "dispatch":
            self.accw[k] += t - self.last_end[k]
        x = self._x(eng, t)
        m = len(x)
        s1 = float(x.sum())
        s2 = float(np.dot(x, x))
        J = s1 * s1 / (m * s2) if s2 > 0 else 1.0
        xbar = s1 / m if m else 0.0
        xk = self._xk(k, eng, t)
        beta = max(0.0, (xbar - xk) / xbar) if xbar > 0 else 0.0
        if self.mode == "full":
            boost = (1.0 - J) * beta * self.sf
        elif self.mode == "jfi_only":
            boost = (1.0 - J) * self.sf
        else:
            boost = beta * self.sf
        self.boost_sum += boost
        self.boost_n += 1
        self.beta_one += beta > 0.999
        self.last = dict(x_k=xk, xbar=xbar, beta=beta, J=J, boost=boost)
        return eng.tq_base * (1.0 + boost)

    def after_run(self, k, ran, eng):
        if self.waiting == "dispatch":
            self.last_end[k] = eng.t

    def info(self):
        n = max(1, self.boost_n)
        return dict(mean_boost=self.boost_sum / n, frac_beta1=self.beta_one / n)


class RandomBoostRR(Policy):
    """Control: q = TQ_base * (1 + U(0, 2*mean_boost)) - same average boost as a given
    EED-RR run, but untargeted. Tests whether *targeting* matters."""

    def __init__(self, mean_boost: float, seed: int = 0, label: str = "RandomBoost"):
        self.mb, self.seed, self.name = float(mean_boost), seed, label

    def reset(self, eng):
        self.rng = np.random.default_rng(self.seed)
        self.boost_sum, self.boost_n = 0.0, 0

    def quantum(self, k, eng):
        boost = self.rng.uniform(0.0, 2.0 * self.mb)
        self.boost_sum += boost
        self.boost_n += 1
        return eng.tq_base * (1.0 + boost)

    def info(self):
        return dict(mean_boost=self.boost_sum / max(1, self.boost_n))
