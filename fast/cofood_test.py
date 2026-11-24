"""
Direct test of shared-food coordination (sparse world v2).

From each shared-food-ON run's own checkpoint, run the WHOLE population forward
2,000 ticks twice: normally, and with shared food invisible to the agents (it
still appears and can still be eaten). Same agents, same world, same random
stream. If agents coordinate on shared food, hiding it should cut how much gets
eaten. Done at tick 10,000 (early) and 100,000 (late): a learned skill shows
up as a bigger normal-vs-blind gap late than early.

    python fast/cofood_test.py
"""
import copy, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
from engine import FastSim
from modes import MODES
m = MODES["sparse2_cofood"]
T = 2000
rows = []
for s in [1, 2, 3, 4, 5]:
    for tick in (10000, 100000):
        p = f"runs/{m['name']}/{m['prefix']}_on_seed{s}/checkpoint_{tick}.pkl"
        if not os.path.exists(p):
            continue
        res = {}
        for cond in ("normal", "blind"):
            sim = FastSim.from_checkpoint(p)          # identical start, identical random stream
            sim.cfg = dict(sim.cfg, cofood_visible=(cond == "normal"))
            before = dict(sim.cofood_stats)
            sim.run(T, extinction_stops=False)
            d = {k: sim.cofood_stats[k] - before[k] for k in before}
            res[cond] = (d["eaten"] / max(d["spawned"], 1), d["solo_attempts"] / max(d["spawned"], 1), sim.population)
        rows.append((s, tick, res))
        print(f"seed {s} tick {tick:>6}: eaten share normal {res['normal'][0]:.2f} vs blind {res['blind'][0]:.2f} "
              f"(gap {res['normal'][0] - res['blind'][0]:+.2f}) | solo tries per item {res['normal'][1]:.1f} vs {res['blind'][1]:.1f} "
              f"| pop {res['normal'][2]} vs {res['blind'][2]}", flush=True)
for tick in (10000, 100000):
    g = [r[2]["normal"][0] - r[2]["blind"][0] for r in rows if r[1] == tick]
    if g:
        print(f"tick {tick}: mean gap (normal - blind) {np.mean(g):+.3f}, normal higher on {sum(x > 0 for x in g)}/{len(g)} seeds")
