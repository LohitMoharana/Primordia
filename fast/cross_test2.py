"""Same world, same starting energy (5) and age (100): compare brains only.
Reports energy gained over 150 steps and eat-fraction, ON minus OFF."""
import sys, numpy as np
import evaluate_fast as v
from modes import MODES
mode = sys.argv[1]; tick = int(sys.argv[2]); seeds = [int(x) for x in sys.argv[3:]] or range(1, 11)
m = MODES[mode]
P = lambda a, s: f"runs/{m['name']}/{m['prefix']}_{a}_seed{s}/checkpoint_{tick}.pkl"
res = {w: {"energy": [], "eat": []} for w in ("off", "on")}
for s in seeds:
    line = f"seed {s:>2}:"
    for world in ("off", "on"):
        try:
            on = v.profile(P("on", s), world_path=P(world, s), reset_state=True)
            off = v.profile(P("off", s), world_path=P(world, s), reset_state=True)
        except FileNotFoundError:
            line += f"  {world} world n/a"; continue
        de, dt = on["energy"] - off["energy"], on["eat"] - off["eat"]
        res[world]["energy"].append(de); res[world]["eat"].append(dt)
        line += f"  {world.upper()} world: energy {on['energy']:+.2f} vs {off['energy']:+.2f} ({de:+.2f}), eat {dt:+.3f}"
    print(line, flush=True)
for w, r in res.items():
    if r["energy"]:
        e = np.array(r["energy"]); a = np.array(r["eat"])
        print(f"{w.upper()} world: energy diff mean {e.mean():+.2f} (ON better {int((e > 0).sum())}/{len(e)}), eat diff mean {a.mean():+.3f} (ON higher {int((a > 0).sum())}/{len(a)})")
