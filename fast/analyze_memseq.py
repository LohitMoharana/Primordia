"""
Memory under pressure (runs/92_memseq_mem, runs/93_memseq_nomem).

    python fast/analyze_memseq.py            # summary of all 4 arms
    python fast/analyze_memseq.py --tests    # + memory lesion on memory runs

Arms: memory on/off x stepping stones on/off. Sequence: dig, stay, eat on one buried-food cell.
Summary from steps.csv (last 10 records): completed sequences per 1,000 ticks, per creature, and
the tick when completions first reached 100 per 1,000 ticks.

--tests (memory runs, final save): the same creatures (energy 5, age 0) for 300 ticks with
(a) memory intact, (b) memory switched off (hidden state reset to zero every tick); completed
sequences per creature. If memory is how they keep track of the step, (b) should be lower.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def summary(run):
    S = list(csv.DictReader(open(os.path.join(run, "steps.csv"))))
    last = S[-10:]
    comp = np.mean([float(r["completed_sequences"]) for r in last])
    pop = np.mean([float(r["population"]) for r in last])
    first = next((int(r["tick"]) for r in S if float(r["completed_sequences"]) >= 100), None)
    return dict(run=os.path.basename(run), tick=int(S[-1]["tick"]), died=os.path.exists(os.path.join(run, "EXTINCT")),
                comp=comp, per=comp / max(pop, 1), pop=pop, first=first)


def lesion(ckpt, no_memory, ticks=300, rep=0):
    from ways import WaysSim
    s = WaysSim.from_checkpoint(ckpt)
    s.out_dir = None
    s.rng = np.random.default_rng(900 + rep)
    s.rng_ways = np.random.default_rng(901 + rep)
    idx = np.flatnonzero(s.alive)
    s.energy[idx] = 5.0
    s.age[idx] = 0
    n0 = len(idx)
    s._steps_stats = dict(completed=0, partial=0, dug_partial=0.0, dug_complete=0.0)
    for _ in range(ticks):
        if no_memory:
            s.hstate[:] = 0.0
        s.step()
    return s._steps_stats["completed"] / n0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    a = ap.parse_args()
    rows = []
    for folder, mem in (("92_memseq_mem", True), ("93_memseq_nomem", False)):
        for r in sorted(glob.glob(os.path.join(HERE, "runs", folder, "memseq_*"))):
            if os.path.exists(os.path.join(r, "steps.csv")):
                s = summary(r)
                s.update(memory=mem, stones="_on_" in s["run"], path=r)
                rows.append(s)
    for s in rows:
        print(f"{'memory' if s['memory'] else 'no memory'} {'+ stones' if s['stones'] else 'no stones'} {s['run']}: "
              f"{'DIED ' if s['died'] else ''}tick {s['tick']} completions/1k {s['comp']:.0f} per creature {s['per']:.2f} "
              f"pop {s['pop']:.0f} first>=100/1k {s['first']}")
    print()
    for mem in (True, False):
        for st in (True, False):
            g = [s for s in rows if s["memory"] == mem and s["stones"] == st and not s["died"]]
            if g:
                ev = [s for s in g if s["comp"] >= 100]
                print(f"{'memory   ' if mem else 'no memory'} {'stones   ' if st else 'no stones'}: evolved {len(ev)}/{len(g)}, "
                      f"completions per creature per 1k ticks {np.mean([s['per'] for s in g]):.2f}, "
                      f"first >=100/1k median {np.median([s['first'] for s in ev]) if ev else '-'}")
    if a.tests:
        print()
        for s in rows:
            if not s["memory"] or s["died"]:
                continue
            ck = sorted(glob.glob(os.path.join(s["path"], "checkpoint_*.pkl")), key=lambda p: int(p.split("_")[-1][:-4]))
            if ck:
                w, wo = lesion(ck[-1], False), lesion(ck[-1], True)
                print(f"{s['run']} ({'stones' if s['stones'] else 'no stones'}): completions per creature in 300 ticks, "
                      f"memory {w:.3f}, memory off {wo:.3f}", flush=True)
