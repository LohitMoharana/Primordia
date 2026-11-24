"""Fair comparison: test ON-arm and OFF-arm agents in the SAME world snapshot.
For each seed: both arms' agents in the OFF world, then both in the ON world."""
import sys, numpy as np
import evaluate_fast as v
from modes import MODES
mode = sys.argv[1]; tick = int(sys.argv[2]); seeds = [int(x) for x in sys.argv[3:]] or range(1, 11)
m = MODES[mode]
import os
RESET = os.environ.get('RESET', '1') == '1'
P = lambda a, s: f"runs/{m['name']}/{m['prefix']}_{a}_seed{s}/checkpoint_{tick}.pkl"
rows = {"off": [], "on": []}
for s in seeds:
    line = f"seed {s:>2}:"
    for world in ("off", "on"):
        try:
            on = v.profile(P("on", s), world_path=P(world, s), reset_state=RESET); off = v.profile(P("off", s), world_path=P(world, s), reset_state=RESET)
        except FileNotFoundError:
            line += f"  {world} world: n/a"; continue
        d = on["survival"] - off["survival"]; rows[world].append(d)
        line += f"  in {world.upper()} world: ON {on['survival']:5.1f} OFF {off['survival']:5.1f} diff {d:+6.1f}"
    print(line, flush=True)
for w, r in rows.items():
    if r: print(f"{w.upper()} world: mean diff {np.mean(r):+.1f}, ON agents better on {sum(x > 0 for x in r)}/{len(r)}")
