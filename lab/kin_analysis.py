"""
Do agents favour kin? From a checkpoint, run sampled agents in their world
(all agents present, so neighbours are real) and record, at every tick:
  - how related the agent's neighbours are (mean kin similarity within radius 1)
  - whether it chose to share
Reports the share rate when relatives are near vs when they are not, and how
related actual share recipients were.

    python3 kin_analysis.py <code_dir> <checkpoint> [ticks]
"""
import copy
import os
import pickle
import sys

import numpy as np

code, ckpt = sys.argv[1], sys.argv[2]
ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 60
sys.path.insert(0, code)
os.chdir(code)

with open(ckpt, "rb") as f:
    data = pickle.load(f)
world, agents = copy.deepcopy(data["world"]), copy.deepcopy(data["agents"])
rng = np.random.default_rng(13)

records = []  # (neighbour relatedness, shared?)
gifts = []
for _ in range(ticks):
    world.step()
    world.set_occupancy(agents)
    world.set_signals(agents)
    rel = {}
    for a in agents:
        nb = [b for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              for b in world.agents_by_cell.get(world.wrap(a.x + dx, a.y + dy), []) if b is not a]
        rel[a.id] = float(np.mean([np.exp(-np.linalg.norm(b.tag - a.tag)) for b in nb])) if nb else None
    new = []
    for a in list(agents):
        if not a.alive:
            continue
        a.last_share_similarity = None
        child = a.step(world, rng)
        if rel.get(a.id) is not None:
            records.append((rel[a.id], a.last_action == "share"))
        if getattr(a, "last_share_similarity", None) is not None:
            gifts.append(a.last_share_similarity)
        if child:
            new.append(child)
    agents = [a for a in agents if a.alive] + new

r = np.array(records, dtype=float)
hi = r[r[:, 0] >= np.median(r[:, 0])]
lo = r[r[:, 0] < np.median(r[:, 0])]
print(f"checkpoint {ckpt}: {len(r)} agent-ticks with neighbours over {ticks} ticks")
print(f"share rate with closely related neighbours: {hi[:, 1].mean():.2%}")
print(f"share rate with distant neighbours:         {lo[:, 1].mean():.2%}")
print(f"ratio (close / distant): {hi[:, 1].mean() / max(lo[:, 1].mean(), 1e-9):.2f}  (1.0 = no kin preference)")
if gifts:
    print(f"mean relatedness of recipients: {np.mean(gifts):.3f}  vs of all neighbours: {r[:, 0].mean():.3f}")
