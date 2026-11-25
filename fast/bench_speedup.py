"""
Speed check for the engine speed-ups (25 Nov 2026).

    python fast/bench_speedup.py

1. One world at 1, 2, 4, 8 and all threads (results are identical at any thread count;
   this only measures speed).
2. Arms-race world: compiled predators vs the old Python predator code.
Uses saved worlds from runs/35_terrain_language and runs/40_arms_race.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import FastSim  # noqa: E402
from predators import PredSim  # noqa: E402

RUNS = os.path.join(HERE, "runs")


def speed(sim, ticks=1500, warm=150):
    for _ in range(warm):
        sim.step()
    t = time.perf_counter()
    for _ in range(ticks):
        sim.step()
    return ticks / (time.perf_counter() - t)


def load(cls, rel):
    s = cls.from_checkpoint(os.path.join(RUNS, rel))
    s.out_dir = None
    return s


if __name__ == "__main__":
    import numba
    terrain = "35_terrain_language/language_on_seed1/checkpoint_1000000.pkl"
    arms_off = "40_arms_race/predevolve_off_seed1/checkpoint_1000000.pkl"
    print(f"{numba.config.NUMBA_NUM_THREADS} threads available")
    counts = sorted({1, 2, 4, 8, numba.config.NUMBA_NUM_THREADS})
    for name, cls, rel in (("terrain world", FastSim, terrain), ("arms-race world", PredSim, arms_off)):
        for n in counts:
            s = load(cls, rel)
            s.set_threads(n)
            print(f"{name}, {n:2d} threads: {speed(s):6.0f} ticks/s (population {s.population})", flush=True)
    for kern in ("python", "numba"):
        s = load(PredSim, arms_off)
        s.set_threads(1)
        s.cfg["pred_kernel"] = kern
        print(f"arms-race world, 1 thread, {kern:6s} predators: {speed(s):6.0f} ticks/s", flush=True)
