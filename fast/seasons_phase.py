"""Seasons survival at other points of the cycle: continue each run from tick
25,000 and test both arms at the same later ticks."""
import os, sys, tempfile, numpy as np
from engine import FastSim
import evaluate_fast as v
extra = [150, 300, 450]   # ticks after 25,000: a quarter, half, three quarters of the 600-tick cycle
res = {e: [] for e in extra}
for s in range(1, 11):
    sims = {a: FastSim.from_checkpoint(f"runs/03_seasons/seasons_{a}_seed{s}/checkpoint_25000.pkl") for a in ("on", "off")}
    done = 0
    for e in extra:
        prof = {}
        for a, sim in sims.items():
            sim.run(e - done, extinction_stops=False)
            d = tempfile.mkdtemp(); sim.out_dir = d; sim.checkpoint()
            prof[a] = v.profile(os.path.join(d, f"checkpoint_{sim.tick}.pkl"))
            sim.out_dir = None
        done = e
        res[e].append(prof["on"]["survival"] - prof["off"]["survival"])
    print(f"seed {s:>2}: " + "  ".join(f"+{e}: {res[e][-1]:+6.1f}" for e in extra), flush=True)
p = lambda t: np.sin(2 * np.pi * t / 600)
for e in extra:
    print(f"tick {25000+e} (regrowth factor {1+0.9*p(25000+e):.2f}): mean {np.mean(res[e]):+.1f}, ON better on {sum(x > 0 for x in res[e])}/10")
