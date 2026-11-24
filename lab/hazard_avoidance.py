"""
Do agents avoid poison? Share of agent-ticks spent on a poison cell vs the
share of the grid that is poison. A random walker lands on poison about as
often as it covers the grid; well below that means agents sense and avoid it.

    python3 hazard_avoidance.py <code_dir> <checkpoint> [n_agents] [steps]
"""
import copy
import os
import pickle
import sys

import numpy as np

code, ckpt = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 150
sys.path.insert(0, code)
os.chdir(code)

with open(ckpt, "rb") as f:
    data = pickle.load(f)
world, agents = data["world"], data["agents"]
rng = np.random.default_rng(11)
sample = rng.choice(agents, size=min(n, len(agents)), replace=False)

on_poison = total = 0
cover = []
for a in sample:
    w, ag = copy.deepcopy(world), copy.deepcopy(a)
    r = np.random.default_rng(int(ag.id) + 5)
    for _ in range(steps):
        w.step()
        ag.step(w, r)
        if not ag.alive:
            break
        cover.append(w.hazard.mean())
        total += 1
        on_poison += int(w.hazard[ag.x, ag.y] > 0)

share, base = on_poison / max(total, 1), float(np.mean(cover)) if cover else 0.0
print(f"checkpoint: {ckpt}  agents: {len(sample)}  agent-ticks: {total}")
print(f"time on poison: {share:.2%}   poison cover: {base:.2%}   ratio: {share / base if base else float('nan'):.2f}")
print("ratio well below 1 = avoidance; about 1 = ignoring it; above 1 = drawn to it")
