"""
Do agents keep away from predators? Runs the WHOLE world (predators moving
and hunting, every agent acting) from a checkpoint for N ticks in a scratch
copy, and compares:
  - share of agent-ticks spent within 1 cell (Chebyshev) of a predator
  - share of grid cells within 1 cell of a predator (where a predator-blind
    agent spread over the grid would land)
Ratio well below 1 = avoidance; about 1 = ignoring predators.

    python3 predator_avoidance.py <code_dir> <checkpoint> [ticks]
"""
import os
import shutil
import sys
import tempfile

import numpy as np

code, ckpt = sys.argv[1], sys.argv[2]
ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 100
sys.path.insert(0, code)
os.chdir(code)
from sim import Simulation  # noqa: E402

scratch = tempfile.mkdtemp(prefix="predavoid_")
try:
    sim = Simulation.from_checkpoint(ckpt, scratch)
    near_agent = total_agent = 0
    near_cell = []
    for _ in range(ticks):
        sim.step()
        w = sim.world
        if not w.pred_pos or not sim.agents:
            continue
        danger = np.zeros((w.size, w.size), bool)
        for px, py in w.pred_pos:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    danger[(px + dx) % w.size, (py + dy) % w.size] = True
        near_cell.append(danger.mean())
        for a in sim.agents:
            total_agent += 1
            near_agent += int(danger[a.x, a.y])
    share, base = near_agent / max(total_agent, 1), float(np.mean(near_cell)) if near_cell else 0.0
    print(f"checkpoint {ckpt}: {ticks} ticks, {total_agent} agent-ticks, kills so far {sim.world.kills}")
    print(f"agents near a predator: {share:.2%}   cells near a predator: {base:.2%}   "
          f"ratio: {share / base if base else float('nan'):.2f}")
    print("ratio well below 1 = avoidance; about 1 = ignoring; above 1 = predators find crowds")
finally:
    shutil.rmtree(scratch)
