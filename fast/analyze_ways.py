"""
Several ways to live: did the population split into surface eaters and diggers?

    python fast/analyze_ways.py                 # all runs in runs/50_ways
    python fast/analyze_ways.py --tests         # also behaviour + lesion tests (slower)

From ways.csv (every 1,000 ticks): gut distribution, share of surface specialists
(g < 1/3) and diggers (g > 2/3), share of actions that are digs.
A run counts as SPLIT when, averaged over its last 10 records, at least 10% of the
population are surface specialists AND at least 10% are diggers.

--tests, on each run's final save (ON arm):
  behaviour  share of digs among each group's actions over 300 ticks;
  lesions    the same creatures (energy reset to 5) for 300 ticks with
             (a) nothing changed, (b) digging yields nothing, (c) blind to buried food;
             score = mean final energy, counting the dead as 0, per gut group.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "runs", "50_ways")


def read(path):
    return list(csv.DictReader(open(path)))


def summary(run):
    rows = read(os.path.join(run, "ways.csv"))
    last = rows[-10:]
    f = lambda k: float(np.mean([float(r[k]) for r in last]))
    surf, dig = f("share_surface_specialists"), f("share_diggers")
    return dict(run=os.path.basename(run), tick=int(rows[-1]["tick"]), pop=f("population"), gut=f("mean_gut"),
                surface=surf, diggers=dig, dig_actions=f("dig_share_of_actions"),
                split=surf >= 0.10 and dig >= 0.10,
                first_diggers=next((int(r["tick"]) for r in rows if float(r["share_diggers"]) >= 0.10), None))


def groups(sim):
    idx = np.flatnonzero(sim.alive)
    g = sim.gut[idx]
    return {"surface (g<1/3)": idx[g < 1 / 3], "middle": idx[(g >= 1 / 3) & (g <= 2 / 3)], "diggers (g>2/3)": idx[g > 2 / 3]}


def behaviour(ckpt, ticks=300):
    from engine import A
    from ways import WaysSim
    s = WaysSim.from_checkpoint(ckpt)
    s.out_dir = None
    G = {k: set(s.aid[v].tolist()) for k, v in groups(s).items()}
    cnt = {k: np.zeros(2) for k in G}
    for _ in range(ticks):
        s.step()
        idx = np.flatnonzero(s.alive)
        la, aid = s.last_action[idx], s.aid[idx]
        for k, ids in G.items():
            m = np.fromiter((a in ids for a in aid), bool, len(aid))
            cnt[k] += [(la[m] == A["mark"]).sum(), m.sum()]
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in cnt.items()}


def lesion(ckpt, kind, ticks=300, rep=0):
    from ways import WaysSim
    s = WaysSim.from_checkpoint(ckpt)
    s.out_dir = None
    s.rng = np.random.default_rng(500 + rep)
    s.rng_ways = np.random.default_rng(600 + rep)
    idx = np.flatnonzero(s.alive)
    s.energy[idx] = 5.0
    G = groups(s)
    ids = {k: s.aid[v].copy() for k, v in G.items()}
    if kind == "no_dig":
        s._dig_eff = lambda i: np.zeros(len(i))
    elif kind == "blind":
        s._mark_slot_args = lambda: (np.zeros((s.size, s.size)), False)
    for _ in range(ticks):
        s.step()
    alive_ids = dict(zip(s.aid[s.alive].tolist(), s.energy[s.alive].tolist()))
    return {k: (float(np.mean([alive_ids.get(int(a), 0.0) for a in v])) if len(v) else float("nan"))
            for k, v in ids.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(a.runs, "*")) if os.path.exists(os.path.join(d, "ways.csv")))
    S = [summary(r) for r in runs]
    print("run | tick | population | mean gut | surface specialists | diggers | digs/actions | split | diggers >=10% from")
    for s in S:
        print(f"{s['run']} | {s['tick']} | {s['pop']:.0f} | {s['gut']:.2f} | {s['surface']:.2f} | {s['diggers']:.2f} | "
              f"{s['dig_actions']:.3f} | {'SPLIT' if s['split'] else '-'} | {s['first_diggers']}")
    for arm in ("on", "off"):
        ss = [s for s in S if f"_{arm}_" in s["run"]]
        if ss:
            print(f"{arm}: split on {sum(s['split'] for s in ss)}/{len(ss)}, mean gut {np.mean([s['gut'] for s in ss]):.2f}, "
                  f"diggers {np.mean([s['diggers'] for s in ss]):.2f}, digs/actions {np.mean([s['dig_actions'] for s in ss]):.3f}")
    if a.tests:
        for r in runs:
            if "_on_" not in os.path.basename(r):
                continue
            ck = sorted(glob.glob(os.path.join(r, "checkpoint_*.pkl")), key=lambda p: int(p.split("_")[-1][:-4]))
            if not ck:
                continue
            c = ck[-1]
            b = behaviour(c)
            L = {k: lesion(c, k) for k in ("none", "no_dig", "blind")}
            print(f"\n{os.path.basename(r)} ({os.path.basename(c)})")
            for gname in b:
                print(f"  {gname:16s} digs/actions {b[gname]:.3f} | energy: normal {L['none'][gname]:.2f}, "
                      f"no digging {L['no_dig'][gname]:.2f}, blind to buried food {L['blind'][gname]:.2f}")
