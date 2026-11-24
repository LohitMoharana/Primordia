"""
Analysis of the long sparse-world-v2 experiments (100,000 ticks, seeds 1-5).

    python fast/analyze_sparse2.py            (from the Primordia folder)

Lesion tests (memory / communication / language): the ON arm's own agents, in
their own world, equal energy and age, with vs without the feature, at ticks
90,000 and 100,000. Language: information in the second symbol (ON vs OFF).
Shared food: co-eating over time, controlled for population. Novelty meter:
how much behaviour changes per 1,000 ticks early vs late, lineages alive.
Writes fast/sparse2_results.md as well as printing.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import evaluate_fast as v
import analyze_language as al
from modes import MODES

SEEDS = [1, 2, 3, 4, 5]
READ = [90000, 100000]
out = []
def say(s=""):
    print(s, flush=True); out.append(s)

def P(mode, arm, s, t):
    m = MODES[mode]
    return f"runs/{m['name']}/{m['prefix']}_{arm}_seed{s}/checkpoint_{t}.pkl"

def D(mode, arm, s):
    m = MODES[mode]
    return f"runs/{m['name']}/{m['prefix']}_{arm}_seed{s}"

def late_pop(d):
    rows = [r for r in csv.DictReader(open(os.path.join(d, "log.csv"))) if int(float(r["tick"])) > 80000]
    return np.mean([float(r["population"]) for r in rows]) if rows else float("nan")

def lesion(mode, key):
    say(f"\n## {mode}: lesion test (remove '{key}' from the ON arm's own agents)")
    say("| seed | energy with / without (90k) | (100k) | eat-fraction with / without (100k) | late population ON / OFF |")
    say("|---|---|---|---|---|")
    de, da = [], []
    for s in SEEDS:
        cells = []
        for t in READ:
            if not os.path.exists(P(mode, "on", s, t)):
                cells.append("n/a"); continue
            a = v.profile(P(mode, "on", s, t), reset_state=True)
            b = v.profile(P(mode, "on", s, t), reset_state=True, override={key: False})
            de.append(a["energy"] - b["energy"]); da.append(a["eat"] - b["eat"])
            cells.append(f"{a['energy']:+.2f} / {b['energy']:+.2f}")
            last = (a, b)
        eat = f"{last[0]['eat']:.3f} / {last[1]['eat']:.3f}" if len(cells) and cells[-1] != "n/a" else "n/a"
        say(f"| {s} | {cells[0]} | {cells[1]} | {eat} | {late_pop(D(mode, 'on', s)):.0f} / {late_pop(D(mode, 'off', s)):.0f} |")
    de, da = np.array(de), np.array(da)
    if len(de):
        say(f"Keeping {key}: energy {de.mean():+.2f} (better on {int((de > 0).sum())}/{len(de)} readings), "
            f"eat-fraction {da.mean():+.3f} (higher on {int((da > 0).sum())}/{len(da)})")

def language_info():
    say("\n## sparse2_language: information in the second symbol (bits above shuffled baseline, tick 100,000)")
    say("| seed | arm | P(signal) | food here: sym1 / sym2 adds | cluster direction: sym1 / sym2 adds |")
    say("|---|---|---|---|---|")
    add = {"on": [], "off": []}
    for s in SEEDS:
        for arm in ("on", "off"):
            p = P("sparse2_language", arm, s, 100000)
            if not os.path.exists(p):
                continue
            r = al.analyse(p)
            fh, cd = r["info"]["food here"], r["info"].get("cluster dir", (float("nan"), float("nan")))
            add[arm].append(cd[1])
            say(f"| {s} | {arm.upper()} | {r['p_signal']:.3f} | {fh[0]:+.4f} / {fh[1]:+.4f} | {cd[0]:+.4f} / {cd[1]:+.4f} |")
    for arm, x in add.items():
        if x:
            say(f"{arm.upper()}: second symbol adds {np.mean(x):+.4f} bits about cluster direction on average")

def cofood():
    say("\n## sparse2_cofood: shared food (worth 8, lasts 30 ticks, needs 2 eaters)")
    say("| seed | eaten share 0-10k -> 90-100k | expired | trend after removing population (r) | wasted solo tries per item per 1000 agents |")
    say("|---|---|---|---|---|")
    for s in SEEDS:
        d = D("sparse2_cofood", "on", s)
        if not os.path.exists(os.path.join(d, "cofood.csv")):
            continue
        c = {int(r["tick"]): r for r in csv.DictReader(open(os.path.join(d, "cofood.csv")))}
        L = {int(float(r["tick"])): float(r["population"]) for r in csv.DictReader(open(os.path.join(d, "log.csv")))}
        ts = sorted(c)
        rows = []
        for lo in range(0, 100000, 2000):
            a = max([t for t in ts if t <= lo], default=None)
            bs = [t for t in ts if t <= lo + 2000]
            if not bs: continue
            b = max(bs)
            g = lambda k: int(c[b][k]) - (int(c[a][k]) if a else 0)
            sp = g("spawned")
            if sp <= 0: continue
            pop = np.mean([L[t] for t in L if lo < t <= lo + 2000] or [np.nan])
            rows.append((lo, g("eaten") / sp, g("expired") / sp, g("solo_attempts") / sp / pop * 1000, pop))
        R = np.array(rows)
        if len(R) < 6: continue
        fit = np.polyfit(R[:, 4], R[:, 1], 1)
        res = R[:, 1] - np.polyval(fit, R[:, 4])
        say(f"| {s} | {R[:5, 1].mean():.2f} -> {R[-5:, 1].mean():.2f} | {R[:5, 2].mean():.2f} -> {R[-5:, 2].mean():.2f} | "
            f"{np.corrcoef(res, R[:, 0])[0, 1]:+.2f} | {R[:5, 3].mean():.2f} -> {R[-5:, 3].mean():.2f} |")

def novelty():
    say("\n## Novelty meter (all runs): behaviour change per 1,000 ticks (L1, 0-2), early vs late; lineages alive at the end")
    say("| experiment | arm | change, ticks 2-20k | change, 80-100k | lineages at end (5 seeds) |")
    say("|---|---|---|---|---|")
    for mode in ("sparse2_memory", "sparse2_comm", "sparse2_language", "sparse2_cofood"):
        for arm in ("on", "off"):
            e, l, lin = [], [], []
            for s in SEEDS:
                f = os.path.join(D(mode, arm, s), "novelty.csv")
                if not os.path.exists(f): continue
                rows = [r for r in csv.DictReader(open(f)) if r["change_from_previous_window"]]
                e += [float(r["change_from_previous_window"]) for r in rows if int(r["tick"]) <= 20000]
                l += [float(r["change_from_previous_window"]) for r in rows if int(r["tick"]) > 80000]
                if rows: lin.append(int(rows[-1]["lineages_alive"]))
            if e:
                say(f"| {mode} | {arm.upper()} | {np.mean(e):.3f} | {np.mean(l) if l else float('nan'):.3f} | {', '.join(map(str, lin))} |")

if __name__ == "__main__":
    say("# Sparse world v2: 100,000-tick experiments, seeds 1-5")
    lesion("sparse2_memory", "memory")
    lesion("sparse2_comm", "communication")
    lesion("sparse2_language", "language")
    language_info()
    cofood()
    novelty()
    open("sparse2_results.md", "w").write("\n".join(out) + "\n")
