"""Cross-seed analysis of fast-engine stage-mode runs (fast/runs/<stage>/...).
Two readings per seed, both arms at the SAME tick (so seasons are phase-matched)."""
import glob, os, pickle, sys, numpy as np
import evaluate_fast as v
from modes import MODES

def final_pop(d):
    ck = sorted(glob.glob(os.path.join(d, "checkpoint_*.pkl")), key=lambda f: int(f.split("_")[-1][:-4]))
    if not ck:
        return 0, 0
    with open(ck[-1], "rb") as f:
        x = pickle.load(f)
    return int(x["tick"]), len(x["x"])

def analyse(mode, seeds):
    m = MODES[mode]
    T = m["ticks"]
    reads = [T - 2500, T]
    out = []
    for s in seeds:
        dirs = {a: os.path.join("runs", m["name"], f"{m['prefix']}_{a}_seed{s}") for a in ("on", "off")}
        fin = {a: final_pop(d) for a, d in dirs.items()}
        alive = {a: fin[a][1] > 0 and fin[a][0] >= T for a in dirs}
        row = dict(seed=s, extinct=[a for a in dirs if not alive[a]])
        if all(alive.values()):
            for i, t in enumerate(reads):
                on = v.profile(os.path.join(dirs["on"], f"checkpoint_{t}.pkl"))
                off = v.profile(os.path.join(dirs["off"], f"checkpoint_{t}.pkl"))
                row[f"surv{i}"] = on["survival"] - off["survival"]
                row[f"eat{i}"] = on["eat"] - off["eat"]
                mv = lambda r: sum(r["action_share"].get(k, 0) for k in "NSEW")
                row[f"move{i}"] = (mv(on), mv(off))
                row[f"rep{i}"] = (on["action_share"].get("reproduce", 0), off["action_share"].get("reproduce", 0))
                row[f"poison{i}"] = on["poison_ratio"]
                row[f"pop{i}"] = (on["population"], off["population"])
        out.append(row)
    return out

if __name__ == "__main__":
    mode = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or list(range(1, 11))
    for r in analyse(mode, seeds):
        if r["extinct"]:
            print(f"seed {r['seed']:>2}: extinct arm(s) {r['extinct']}")
            continue
        print(f"seed {r['seed']:>2}: survival {r['surv0']:+6.1f} {r['surv1']:+6.1f} | eat {r['eat0']:+.3f} {r['eat1']:+.3f} | "
              f"move ON/OFF {r['move1'][0]*100:.1f}/{r['move1'][1]*100:.1f}% | reproduce ON/OFF {r['rep1'][0]*100:.1f}/{r['rep1'][1]*100:.1f}% | "
              f"poison ratio {r['poison0']:.2f} {r['poison1']:.2f} | pop {r['pop1'][0]}/{r['pop1'][1]}")
