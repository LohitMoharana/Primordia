"""
Bedau's evolutionary activity statistics from runs recorded with modes_compare.py --bedau.

    python fast/bedau_analyze.py fast/runs/long_run

For every run: each component's activity = its count summed over snapshots (how widely and how long
it has lasted). The threshold comes from the run's own neutral shadow: the highest activity any shadow component
(born after the start) ever reaches, so by construction no chance component passes it. Components
present at the start (the founders') are left out of both. Real components that pass it lasted longer
and wider than anything chance produced: they are counted as adaptive; "new evolutionary activity" is the number that pass it for the first
time in each window.

Reported per run and per world (first half vs second half of the run):
  new_activity_rate   adaptive components appearing per 100k ticks (real), and the same for the shadow
  diversity           components present per snapshot
  mean_activity       summed activity of present components / diversity
Bedau, Snyder & Packard's classes: unbounded evolutionary activity (class 3) shows new activity that
keeps arriving at a steady or rising rate, with growing diversity; bounded (class 2) shows new
activity drying up. Here "unbounded so far" = second-half rate >= 80% of the first-half rate and
well above the shadow.
Output: <dir>/bedau_summary.csv and printed table.
"""
import csv
import glob
import gzip
import json
import os
import sys

import numpy as np


def unpack(z, which):
    ids, cnt, off = z[f"{which}_ids"], z[f"{which}_counts"], z[f"{which}_off"]
    return [(ids[off[i]:off[i + 1]], cnt[off[i]:off[i + 1]]) for i in range(len(off) - 1)]


def activity_series(rows):
    """rows: per snapshot (component ids, counts) -> per snapshot {component: activity so far}."""
    act = {}
    out = []
    for ids, cnt in rows:
        for c, n in zip(ids.tolist(), cnt.tolist()):
            act[c] = act.get(c, 0) + n
        out.append({c: act[c] for c in ids.tolist()})
    return out, act


def analyse(z):
    meta = json.loads(str(z["meta"]))
    rec = dict(world=meta["world"], seed=meta["seed"], mu=z["mu"].tolist(),
               real=unpack(z, "real"), shadow=unpack(z, "shadow"))
    ticks = np.array(z["ticks"])
    real_series, real_final = activity_series(rec["real"])
    sh_series, sh_final = activity_series(rec["shadow"])
    # components already present at the first snapshot (the founders') have a head start in both runs:
    # left out of both. Threshold: the highest activity any later shadow component ever reached.
    born0_r = set(rec["real"][0][0].tolist())
    born0_s = set(rec["shadow"][0][0].tolist())
    later = [a for c, a in sh_final.items() if c not in born0_s]
    theta = float(max(later)) if later else 0.0

    def new_times(series, born0):
        seen, times = set(), []
        for t, row in zip(ticks, series):
            for c, a in row.items():
                if c in born0:
                    continue
                if a > theta and c not in seen:
                    seen.add(c)
                    times.append(t)
        return np.array(times)

    tr, ts = new_times(real_series, born0_r), new_times(sh_series, born0_s)
    T0, T1 = ticks[0], ticks[-1]
    mid = (T0 + T1) / 2
    span = max(1.0, (T1 - T0) / 2) / 100000.0  # half the run, in units of 100k ticks
    div = np.array([len(r[0]) for r in rec["real"]])
    mean_act = np.array([sum(row.values()) / max(1, len(row)) for row in real_series])
    first, second = (tr >= T0) & (tr < mid), tr >= mid
    out = dict(world=rec["world"], seed=rec["seed"], ticks=int(T1), threshold=round(theta, 1),
               adaptive_components=len(tr), shadow_passing=len(ts),
               rate_first_half=round(first.sum() / span, 2), rate_second_half=round(second.sum() / span, 2),
               shadow_rate=round(len(ts) / (2 * span), 2),
               diversity_first_half=round(float(div[ticks < mid].mean()), 2),
               diversity_second_half=round(float(div[ticks >= mid].mean()), 2),
               mean_activity_end=round(float(mean_act[-1]), 1), mu_mean=round(float(np.mean(rec["mu"][1:])), 4))
    r1, r2 = out["rate_first_half"], out["rate_second_half"]
    out["unbounded_so_far"] = bool(r2 >= 0.8 * r1 and r2 > 3 * out["shadow_rate"] and r2 > 0)
    return out


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "bedau", "*.npz")))
    if not files:
        sys.exit(f"no Bedau records in {d}/bedau")
    rows = []
    for fn in files:
        with np.load(fn) as z:
            rows.append(analyse(z))
    with open(os.path.join(d, "bedau_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worlds = []
    for r in rows:
        if r["world"] not in worlds:
            worlds.append(r["world"])
    print("world | runs | adaptive components | per 100k ticks: 1st half, 2nd half, shadow | diversity 1st, 2nd | unbounded so far")
    for wname in worlds:
        rs = [r for r in rows if r["world"] == wname]
        m = lambda k: np.mean([r[k] for r in rs])
        print(f"{wname[:40]} | {len(rs)} | {m('adaptive_components'):.1f} | {m('rate_first_half'):.1f}, "
              f"{m('rate_second_half'):.1f}, {m('shadow_rate'):.2f} | {m('diversity_first_half'):.1f}, "
              f"{m('diversity_second_half'):.1f} | {sum(r['unbounded_so_far'] for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
