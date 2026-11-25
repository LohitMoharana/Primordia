"""
Arms race round 2b (runs/43_arms2b_shelter) and shelter building (runs/70_build).

    python fast/analyze_arms2.py 43_arms2b_shelter
    python fast/analyze_arms2.py 70_build --tests

Per run: survived?, lowest prey count after predators arrive, final prey and predators,
share of prey standing in a shelter vs the share of cells that are shelters (more than
the cell share = prey seek shelters).

--tests (each surviving run's first and last save):
  Red Queen      prey and predators from early (50k) vs late saves, 2x2 (red_queen.table);
  prey senses    kills per predator per 100 ticks with prey blind to predators, and
                 (shelter worlds) blind to shelters, vs normal, at the last save.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def rows(p):
    return list(csv.DictReader(open(p)))


def summary(run):
    P = rows(os.path.join(run, "predators.csv"))
    late = [r for r in P if float(r["tick"]) > 4000]
    out = dict(run=os.path.basename(run), died=os.path.exists(os.path.join(run, "EXTINCT")),
               tick=int(float(P[-1]["tick"])), min_prey=min(int(float(r["prey"])) for r in late) if late else None,
               prey=int(float(P[-1]["prey"])), pred=int(float(P[-1]["predators"])))
    ck = sorted(glob.glob(os.path.join(run, "checkpoint_*.pkl")), key=lambda p: int(p.split("_")[-1][:-4]))
    out["ckpts"] = ck
    if ck and not out["died"]:
        import pickle
        d = pickle.load(open(ck[-1], "rb"))
        sh = d.get("shelter")
        if sh is not None and sh.any():
            out["shelter_cells"] = float((sh > 0).mean())
            out["prey_in_shelter"] = float((sh[d["x"], d["y"]] > 0).mean())
    return out


def blind_shelter_trial(prey_ckpt, pred_ckpt, rep=0):
    import red_queen as q
    from predators import PredSim
    orig = PredSim.step

    def step_blind(self):
        keep = self._has_shelter
        self._has_shelter = False  # prey cannot see shelters (they still protect)
        orig(self)
        self._has_shelter = keep
    PredSim.step = step_blind
    try:
        return q.trial(prey_ckpt, pred_ckpt, rep)
    finally:
        PredSim.step = orig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--early", type=int, default=50000)
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(HERE, "runs", a.folder, "*")) if os.path.exists(os.path.join(d, "predators.csv")))
    S = [summary(r) for r in runs]
    print("run | tick | died | lowest prey | prey | predators | shelter cells | prey in shelters")
    for s in S:
        print(f"{s['run']} | {s['tick']} | {'DIED' if s['died'] else '-'} | {s['min_prey']} | {s['prey']} | {s['pred']} | "
              f"{s.get('shelter_cells', float('nan')):.3f} | {s.get('prey_in_shelter', float('nan')):.3f}")
    for arm in ("on", "off"):
        ss = [s for s in S if f"_{arm}_" in s["run"]]
        if ss:
            print(f"{arm}: survived {sum(not s['died'] for s in ss)}/{len(ss)}")
    if a.tests:
        import red_queen as q
        for s in S:
            if s["died"] or not s["ckpts"]:
                continue
            d = os.path.dirname(s["ckpts"][-1])
            late = int(s["ckpts"][-1].split("_")[-1][:-4])
            early = os.path.join(d, f"checkpoint_{a.early}.pkl")
            if not os.path.exists(early) or late <= a.early:
                continue
            t = q.table(d, a.early, late, reps=2)
            lc = s["ckpts"][-1]
            normal = np.mean([q.trial(lc, lc, r) for r in range(2)])
            blind_p = np.mean([q.blind_trial(lc, lc, r) for r in range(2)])
            line = (f"{s['run']}: early/early {t[(a.early, a.early)]:.2f} early prey vs late predators {t[(a.early, late)]:.2f} "
                    f"late prey vs early predators {t[(late, a.early)]:.2f} late/late {t[(late, late)]:.2f} | "
                    f"kills normal {normal:.2f}, prey blind to predators {blind_p:.2f}")
            if "shelter_cells" in s:
                line += f", blind to shelters {np.mean([blind_shelter_trial(lc, lc, r) for r in range(2)]):.2f}"
            print(line, flush=True)
