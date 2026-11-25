"""
Bigger brains when needed (runs/80_brain_ways, runs/81_brain_arms).

    python fast/analyze_brains.py 80_brain_ways
    python fast/analyze_brains.py 80_brain_ways --tests

From brains.csv (every 1,000 ticks), last 10 records: mean brain size (hidden units in use),
ON (extra units can be wired) vs OFF (extra units never wired: pure cost, so OFF shows
what drift and cost alone do). In the several-ways-to-live world also per gut group.

--tests, each ON run's final save: the same creatures (energy 5, age 0) for 300 ticks,
(a) intact, (b) every unit beyond the founders' 24 switched off (weights zeroed);
score = mean final energy, counting the dead as 0, for creatures with more than 24 units.
If bigger brains are used, (b) should be worse.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def tail(path, n=10):
    return list(csv.DictReader(open(path)))[-n:]


def summary(run):
    R = tail(os.path.join(run, "brains.csv"))
    f = lambda k: float(np.mean([float(r[k]) for r in R if r.get(k) not in ("", None)])) if any(r.get(k) not in ("", None) for r in R) else float("nan")
    return dict(run=os.path.basename(run), tick=int(R[-1]["tick"]), pop=f("population"), units=f("mean_units"),
                umin=f("min_units"), umax=f("max_units"), surf=f("mean_units_surface"), dig=f("mean_units_diggers"),
                died=os.path.exists(os.path.join(run, "EXTINCT")))


def lesion(ckpt, truncate, ticks=300, rep=0):
    from brains import PredBrainSim, WaysBrainSim
    import pickle
    d = pickle.load(open(ckpt, "rb"))
    C = WaysBrainSim if "buried" in d else PredBrainSim
    s = C.from_checkpoint(ckpt)
    s.out_dir = None
    s.rng = np.random.default_rng(800 + rep)
    idx = np.flatnonzero(s.alive)
    s.energy[idx] = 5.0
    s.age[idx] = 0
    big = idx[s.nact[idx] > s.cfg["nact_init"]]
    ids = s.aid[big].copy()
    if truncate and len(big):
        s._zero_units(big, np.full(len(big), s.cfg["nact_init"]))
        s.nact[big] = s.cfg["nact_init"]
    for _ in range(ticks):
        s.step()
    live = dict(zip(s.aid[s.alive].tolist(), s.energy[s.alive].tolist()))
    return float(np.mean([live.get(int(a), 0.0) for a in ids])) if len(ids) else float("nan"), len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(HERE, "runs", a.folder, "*")) if os.path.exists(os.path.join(d, "brains.csv")))
    S = [summary(r) for r in runs]
    print("run | tick | died | population | mean units [min-max] | surface specialists | diggers")
    for s in S:
        print(f"{s['run']} | {s['tick']} | {'DIED' if s['died'] else '-'} | {s['pop']:.0f} | {s['units']:.2f} "
              f"[{s['umin']:.0f}-{s['umax']:.0f}] | {s['surf']:.2f} | {s['dig']:.2f}")
    for arm in ("on", "off"):
        ss = [s for s in S if f"_{arm}_" in s["run"] and not s["died"]]
        if ss:
            print(f"{arm} ({'extra units wired' if arm == 'on' else 'extra units unwired'}): mean units "
                  f"{np.mean([s['units'] for s in ss]):.2f} over {len(ss)} runs; above 24 on {sum(s['units'] > 24 for s in ss)}/{len(ss)}")
    if a.tests:
        for r in runs:
            if "_on_" not in os.path.basename(r) or os.path.exists(os.path.join(r, "EXTINCT")):
                continue
            ck = sorted(glob.glob(os.path.join(r, "checkpoint_*.pkl")), key=lambda p: int(p.split("_")[-1][:-4]))
            if not ck:
                continue
            (i, n), (t, _) = lesion(ck[-1], False), lesion(ck[-1], True)
            print(f"{os.path.basename(r)}: {n} creatures with >24 units: energy intact {i:.2f}, extra units switched off {t:.2f}", flush=True)
