"""Sparse-world experiments: lesion test + same-world fair test, seeds 1-5, two readings.
memory / communication: lesion on the ON arm (does removing it hurt the brains that evolved with it?)
shared food: fair test (both arms in the same world) + cooperation trend."""
import csv, os, sys, numpy as np
import evaluate_fast as v
from modes import MODES

def P(mode, arm, s, t):
    m = MODES[mode]; return f"runs/{m['name']}/{m['prefix']}_{arm}_seed{s}/checkpoint_{t}.pkl"

def lesion(mode, key, seeds, ticks=(22500, 25000)):
    print(f"== {mode}: lesion test ({key} removed from the ON arm's own agents; equal energy/age)")
    de, da = [], []
    for s in seeds:
        parts = []
        for t in ticks:
            if not os.path.exists(P(mode, "on", s, t)):
                parts.append(f"{t}: n/a"); continue
            a = v.profile(P(mode, "on", s, t), reset_state=True)
            b = v.profile(P(mode, "on", s, t), reset_state=True, override={key: False})
            de.append(a["energy"] - b["energy"]); da.append(a["eat"] - b["eat"])
            parts.append(f"{t}: energy {a['energy']:+.2f} vs {b['energy']:+.2f} removed, eat {a['eat']:.3f} vs {b['eat']:.3f}")
        print(f"  seed {s}: " + " | ".join(parts), flush=True)
    de, da = np.array(de), np.array(da)
    if len(de):
        print(f"  -> keeping {key} gives energy {de.mean():+.2f} (better on {int((de > 0).sum())}/{len(de)} readings), eat-fraction {da.mean():+.3f}")

def fair(mode, seeds, t=25000):
    print(f"== {mode}: fair test (both arms in the same world, equal energy/age), tick {t}")
    for w in ("off", "on"):
        de = []
        for s in seeds:
            if not (os.path.exists(P(mode, "on", s, t)) and os.path.exists(P(mode, "off", s, t))):
                continue
            a = v.profile(P(mode, "on", s, t), world_path=P(mode, w, s, t), reset_state=True)
            b = v.profile(P(mode, "off", s, t), world_path=P(mode, w, s, t), reset_state=True)
            de.append(a["energy"] - b["energy"])
        if de:
            de = np.array(de)
            print(f"  in {w.upper()} world: energy ON-OFF {de.mean():+.2f} (ON better on {int((de > 0).sum())}/{len(de)} seeds)")

def population(mode, seeds):
    m = MODES[mode]
    for arm in ("on", "off"):
        pops = []
        for s in seeds:
            f = f"runs/{m['name']}/{m['prefix']}_{arm}_seed{s}/log.csv"
            if os.path.exists(f):
                rows = [r for r in csv.DictReader(open(f)) if int(float(r["tick"])) > 15000]
                pops.append(np.mean([float(r["population"]) for r in rows]) if rows else 0)
        print(f"  {arm.upper()} mean population after tick 15,000: " + ", ".join(f"{p:.0f}" for p in pops))

def cofood_trend(seeds):
    m = MODES["sparse_cofood"]
    for s in seeds:
        f = f"runs/{m['name']}/{m['prefix']}_on_seed{s}/cofood.csv"
        if not os.path.exists(f): continue
        c = {int(r["tick"]): r for r in csv.DictReader(open(f))}
        ts = sorted(c)
        def share(lo, hi):
            a = max([t for t in ts if t <= lo], default=None); b = max(t for t in ts if t <= hi)
            g = lambda k: int(c[b][k]) - (int(c[a][k]) if a else 0)
            return g("eaten") / max(g("spawned"), 1), g("expired") / max(g("spawned"), 1)
        e1, x1 = share(0, 5000); e2, x2 = share(20000, 25000)
        print(f"  seed {s}: shared food eaten {e1:.2f} -> {e2:.2f}, expired uneaten {x1:.2f} -> {x2:.2f}")

if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    lesion("sparse_memory", "memory", seeds); fair("sparse_memory", seeds); population("sparse_memory", seeds)
    lesion("sparse_comm", "communication", seeds); fair("sparse_comm", seeds); population("sparse_comm", seeds)
    fair("sparse_cofood", seeds); population("sparse_cofood", seeds); cofood_trend(seeds)
