"""
Why did shelter worlds die out more often in arms race 2b? (runs/43_arms2b_shelter)

    python fast/why_shelters.py

Uses only the logs and saved worlds (no new simulation):
  1. predators, prey, kills per predator, food on the ground: shelter vs no-shelter worlds
     (each run's average over ticks 50k-200k, before any shelter world died);
  2. the last 20,000 ticks before each collapse;
  3. where prey stand: share inside a shelter, share next to one (within 1 cell),
     against the share of cells that are inside / next to shelters;
  4. prey actions (eat / move / stay).
"""
import csv
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs", "43_arms2b_shelter")


def rows(p):
    return list(csv.DictReader(open(p)))


def window(P, a, b):
    return [r for r in P if a < float(r["tick"]) <= b]


def stats(run, a=50000, b=200000):
    P = rows(os.path.join(run, "predators.csv"))
    W = window(P, a, b)
    if len(W) < 10:
        return None
    pred = np.array([float(r["predators"]) for r in W])
    prey = np.array([float(r["prey"]) for r in W])
    kills = (float(W[-1]["kills"]) - float(W[0]["kills"])) / (len(W) - 1)  # per 1,000 ticks
    L = [r for r in rows(os.path.join(run, "log.csv")) if a < float(r["tick"]) <= b]
    food = np.mean([float(r["total_food"]) for r in L])
    N = window(rows(os.path.join(run, "novelty.csv")), a, b)
    act = lambda cols: float(np.mean([sum(float(r[c]) for c in cols) for r in N]))
    return dict(pred=pred.mean(), prey=prey.mean(), ratio=np.mean(pred / np.maximum(prey, 1)),
                kpp=kills / max(pred.mean(), 1), min_prey=prey.min(), food=food,
                eat=act(["eat"]), move=act(["N", "S", "E", "W"]), stay=act(["stay"]))


def where(run):
    out = []
    for ck in sorted(glob.glob(os.path.join(run, "checkpoint_*.pkl")), key=lambda p: int(p.split("_")[-1][:-4]))[:6]:
        d = pickle.load(open(ck, "rb"))
        sh = d.get("shelter")
        if sh is None or not sh.any():
            return None
        S = sh.shape[0]
        near = sum(np.roll(np.roll(sh, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1)) > 0
        near &= sh == 0
        x, y = d["x"], d["y"]
        out.append(((sh[x, y] > 0).mean(), near[x, y].mean(), (sh > 0).mean(), near.mean()))
    return np.mean(out, 0) if out else None


if __name__ == "__main__":
    runs = sorted(glob.glob(os.path.join(R, "shelter_*_seed*")))
    print("1. ticks 50k-200k, mean per run")
    for arm in ("on", "off"):
        S = [s for s in (stats(r) for r in runs if f"_{arm}_" in r) if s]
        if not S:
            continue
        m = {k: np.mean([s[k] for s in S]) for k in S[0]}
        print(f"  {'shelters' if arm == 'on' else 'none    '} ({len(S)} runs): predators {m['pred']:.0f}, prey {m['prey']:.0f}, "
              f"predators per prey {m['ratio']:.3f}, kills per predator per 1k ticks {m['kpp']:.1f}, lowest prey {m['min_prey']:.0f}, "
              f"food on ground {m['food']:.0f} | prey actions: eat {m['eat']:.3f} move {m['move']:.3f} stay {m['stay']:.3f}")
    print("\n2. last 20,000 ticks before each collapse (every 2,000 ticks: prey / predators)")
    for r in runs:
        if os.path.exists(os.path.join(r, "EXTINCT")):
            P = rows(os.path.join(r, "predators.csv"))
            T = float(P[-1]["tick"])
            W = [p for p in P if float(p["tick"]) > T - 20000][::2]
            print(f"  {os.path.basename(r)} (died {int(T)}): " + "  ".join(f"{int(float(p['prey']))}/{int(float(p['predators']))}" for p in W))
    print("\n3. where prey stand (first saves, shelter worlds): inside shelter vs share of cells; next to one vs share of cells")
    for r in runs:
        if "_on_" in r:
            w = where(r)
            if w is not None:
                print(f"  {os.path.basename(r)}: inside {w[0]:.3f} (cells {w[2]:.3f}, x{w[0] / w[2]:.1f}) | next to {w[1]:.3f} (cells {w[3]:.3f}, x{w[1] / w[3]:.1f})")
