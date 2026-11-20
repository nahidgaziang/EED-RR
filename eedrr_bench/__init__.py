"""EED-RR benchmark: single-CPU scheduling simulator, policies, experiments and statistics."""
from .engine import Engine, Policy, Workload, default_sample_times, jain, simulate
from .policies import (ADRR, EEDRR, FCFS, MARR, RR, SJF, SRTF, EImHLVQTRR, RandomBoostRR,
                       SelfAdjustRR, SmartTimeSliceRR)

__all__ = [
    "Engine", "Policy", "Workload", "default_sample_times", "jain", "simulate",
    "ADRR", "EEDRR", "FCFS", "MARR", "RR", "SJF", "SRTF", "EImHLVQTRR", "RandomBoostRR",
    "SelfAdjustRR", "SmartTimeSliceRR",
]
