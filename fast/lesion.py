"""Lesion test: the ON arm's own agents, in their own world, with equal starting
energy and age, once with the feature's information and once with it removed.
A drop when it is removed means the brains really use it."""
import sys, numpy as np
import evaluate_fast as v
from modes import MODES
mode, tick, key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = MODES[mode]
de, da = [], []
for s in range(1, 11):
    P = f"runs/{m['name']}/{m['prefix']}_on_seed{s}/checkpoint_{tick}.pkl"
    a = v.profile(P, reset_state=True)
    b = v.profile(P, reset_state=True, override={key: False})
    de.append(a["energy"] - b["energy"]); da.append(a["eat"] - b["eat"])
    print(f"seed {s:>2}: with {key}: energy {a['energy']:+.2f} eat {a['eat']:.3f} | removed: energy {b['energy']:+.2f} eat {b['eat']:.3f}", flush=True)
de, da = np.array(de), np.array(da)
print(f"information removed costs: energy {de.mean():+.2f} (worse without it on {int((de > 0).sum())}/10), eat-fraction {da.mean():+.3f} ({int((da > 0).sum())}/10)")
