"""
Are marks useful information? From a checkpoint, run the whole world for N
ticks in a scratch copy and report:
  1. where marks are: average food on marked cells vs unmarked cells
     (marks laid on rich food = "food here" trails)
  2. whether agents follow marks: of an agent's moves made when a neighbouring
     cell is marked more strongly than the others, the share that go INTO that
     strongest-marked neighbour (chance = 25% for 4 directions)
  3. how much marking happens (share of actions)

    python3 marker_analysis.py <code_dir> <checkpoint> [ticks]
"""
import os
import shutil
import sys
import tempfile
from collections import Counter

import numpy as np

code, ckpt = sys.argv[1], sys.argv[2]
ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 60
sys.path.insert(0, code)
os.chdir(code)
from sim import Simulation  # noqa: E402

DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
scratch = tempfile.mkdtemp(prefix="markers_")
try:
    sim = Simulation.from_checkpoint(ckpt, scratch)
    food_marked, food_unmarked = [], []
    follow = chances = 0
    actions = Counter()
    for _ in range(ticks):
        w = sim.world
        before = {a.id: (a.x, a.y) for a in sim.agents}
        marks = w.marker.copy()
        sim.step()
        m = w.marker > 0.5
        if m.any() and (~m).any():
            food_marked.append(w.food[m].mean())
            food_unmarked.append(w.food[~m].mean())
        for a in sim.agents:
            act = getattr(a, "last_action", None)
            actions[act] += 1
            if act in DIRS and a.id in before:
                x0, y0 = before[a.id]
                vals = {d: marks[(x0 + dx) % w.size, (y0 + dy) % w.size] for d, (dx, dy) in DIRS.items()}
                best = max(vals, key=vals.get)
                if vals[best] > 0.5 and sorted(vals.values())[-2] < vals[best]:
                    chances += 1
                    follow += int(act == best)
    total = sum(actions.values()) or 1
    print(f"checkpoint {ckpt}: {ticks} ticks")
    if food_marked:
        print(f"food on marked cells: {np.mean(food_marked):.3f}   on unmarked cells: {np.mean(food_unmarked):.3f}")
    print(f"moves toward the strongest-marked neighbour: {follow}/{chances} = "
          f"{follow / max(chances, 1):.1%}  (chance 25%)")
    print(f"mark action share: {actions.get('mark', 0) / total:.2%}")
finally:
    shutil.rmtree(scratch)
