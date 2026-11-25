"""
Food storage: do creatures evolve to save food and use it later?

    python fast/analyze_storage.py            # all runs in runs/60_storage
    python fast/analyze_storage.py --tests    # also timing + lesion tests (slower)

From storage.csv / novelty.csv (every 1,000 ticks), last 10 records:
  store and take actions as a share of all actions, ON vs OFF (OFF: the same actions do
  nothing, so their share there is what drift and cost alone produce);
  energy stored in the world, and what share of what was taken came back to the
  creature that put it there.

--tests, on each ON run's final save:
  timing   1,800 ticks (3 seasons): store and take rates per creature per tick in the
           good half of the season (food regrowing faster) vs the lean half;
  lesions  same creatures, energy reset to 5 and age to 0, 450 ticks (3/4 of a season):
           (a) nothing changed, (b) storage switched off and stores emptied,
           (c) blind to stores; score = mean final energy, counting the dead as 0.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "runs", "60_storage")


def tail_mean(path, keys, n=10):
    rows = list(csv.DictReader(open(path)))[-n:]
    return {k: float(np.mean([float(r[k] or 0) for r in rows])) for k in keys}


def summary(run):
    nov = tail_mean(os.path.join(run, "novelty.csv"), ["mark", "mate", "eat"])
    out = dict(run=os.path.basename(run), store_share=nov["mark"], take_share=nov["mate"], eat_share=nov["eat"])
    sp = os.path.join(run, "storage.csv")
    if os.path.exists(sp):
        st = tail_mean(sp, ["population", "stored_total", "deposited", "taken", "taken_by_depositor"])
        out.update(pop=st["population"], stored=st["stored_total"], deposited=st["deposited"], taken=st["taken"],
                   own=st["taken_by_depositor"] / st["taken"] if st["taken"] else float("nan"))
    else:
        out.update(pop=tail_mean(os.path.join(run, "log.csv"), ["population"], 100)["population"])
    return out


def season_good(sim):
    c = sim.cfg
    return np.sin(2 * np.pi * sim.t / c["season_period"]) > 0


def timing(ckpt, ticks=1800):
    from engine import A
    from storage import StoreSim
    s = StoreSim.from_checkpoint(ckpt)
    s.out_dir = None
    acc = {True: np.zeros(3), False: np.zeros(3)}  # stores, takes, creature-ticks
    for _ in range(ticks):
        s.step()
        g = bool(season_good(s))
        la = s.last_action[s.alive]
        acc[g] += [(la == A["mark"]).sum(), (la == A["mate"]).sum(), len(la)]
    return {("good" if k else "lean"): (v[0] / v[2], v[1] / v[2]) for k, v in acc.items()}


def lesion(ckpt, kind, ticks=450, rep=0):
    from storage import StoreSim
    s = StoreSim.from_checkpoint(ckpt)
    s.out_dir = None
    s.rng = np.random.default_rng(700 + rep)
    idx = np.flatnonzero(s.alive)
    s.energy[idx] = 5.0
    s.age[idx] = 0  # lifespan is 500 ticks: start everyone young so the test is about energy, not old age
    ids = s.aid[idx].copy()
    if kind == "no_storage":
        s.cfg = dict(s.cfg, storage=False)
        s.store[:] = 0.0
    elif kind == "blind":
        s._mark_slot_args = lambda: (np.zeros((s.size, s.size)), False)
    for _ in range(ticks):
        s.step()
    live = dict(zip(s.aid[s.alive].tolist(), s.energy[s.alive].tolist()))
    return float(np.mean([live.get(int(a), 0.0) for a in ids]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, "*")) if os.path.exists(os.path.join(d, "novelty.csv")))
    S = [summary(r) for r in runs]
    print("run | population | store actions | take actions | stored in world | taken per 1k ticks | taken by depositor")
    for s in S:
        print(f"{s['run']} | {s['pop']:.0f} | {s['store_share']:.4f} | {s['take_share']:.4f} | {s.get('stored', 0):.0f} | "
              f"{s.get('taken', 0):.0f} | {s.get('own', float('nan')):.2f}")
    on = [s for s in S if "_on_" in s["run"]]
    off = [s for s in S if "_off_" in s["run"]]
    if on and off:
        for k, name in (("store_share", "store"), ("take_share", "take")):
            print(f"{name} actions: ON {np.mean([s[k] for s in on]):.4f} vs OFF {np.mean([s[k] for s in off]):.4f}")
        print(f"population: ON {np.mean([s['pop'] for s in on]):.0f} vs OFF {np.mean([s['pop'] for s in off]):.0f}")
    if a.tests:
        for r in runs:
            if "_on_" not in os.path.basename(r):
                continue
            ck = sorted(glob.glob(os.path.join(r, "checkpoint_*.pkl")), key=lambda p: int(p.split("_")[-1][:-4]))
            if not ck:
                continue
            c = ck[-1]
            t = timing(c)
            L = {k: lesion(c, k) for k in ("none", "no_storage", "blind")}
            print(f"\n{os.path.basename(r)} ({os.path.basename(c)})")
            print(f"  store rate good {t['good'][0]:.4f} / lean {t['lean'][0]:.4f} | take rate good {t['good'][1]:.4f} / lean {t['lean'][1]:.4f}")
            print(f"  energy after a season: normal {L['none']:.2f}, no storage {L['no_storage']:.2f}, blind to stores {L['blind']:.2f}")
