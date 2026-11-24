"""
Cross-seed verdict table: does each seed agree with seed 1?

    python lab/summarize.py                       # every stage/seed that has both arms
    python lab/summarize.py --stages 05_hazards 03_seasons

For each (stage, seed) with checkpoints in both arms, runs lab/evaluate.py
(closest-generation ON/OFF pair, 50 agents x 150 steps) and reports the
ON-minus-OFF survival difference. A feature replicates if most seeds point
the same way as seed 1 (see FINDINGS.md for the seed-1 numbers).
Writes lab/summary.csv.
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys

LAB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LAB)
sys.path.insert(0, LAB)
from replicate import STAGES  # noqa: E402


def seeds_with_both_arms(stage):
    prefix = STAGES[stage][0]
    found = {}
    for d in glob.glob(os.path.join(ROOT, "stages", stage, "runs", f"{prefix}_*_seed*")):
        m = re.search(r"_(on|off)_seed(\d+)$", d)
        if m and glob.glob(os.path.join(d, "checkpoint_*.pkl")):
            found.setdefault(int(m.group(2)), set()).add(m.group(1))
    return sorted(s for s, arms in found.items() if arms == {"on", "off"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", choices=sorted(STAGES), default=sorted(STAGES))
    a = ap.parse_args()
    out = []
    for stage in a.stages:
        for seed in seeds_with_both_arms(stage):
            r = subprocess.run([sys.executable, os.path.join(LAB, "evaluate.py"), stage, str(seed)],
                               capture_output=True, text=True, cwd=ROOT)
            surv = pop = gens = None
            for line in r.stdout.splitlines():
                if line.startswith("steps survived"):
                    surv = float(line.split()[-1])
                elif line.startswith("population"):
                    pop = float(line.split()[-1])
                elif line.startswith("Stage"):
                    gens = line.split(":", 1)[1].strip()
            if surv is None:
                print(f"{stage} seed{seed}: evaluate failed\n{r.stderr[-400:]}")
                continue
            verdict = "ON better" if surv > 2 else "OFF better" if surv < -2 else "no difference"
            print(f"{stage:<18} seed{seed}  survival ON-OFF {surv:+7.2f}  pop ON-OFF {pop:+7.0f}  {verdict}  ({gens})")
            out.append([stage, seed, surv, pop, verdict, gens])
    with open(os.path.join(LAB, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "seed", "survival_diff", "population_diff", "verdict", "pair"])
        w.writerows(out)
    print(f"\nwrote {os.path.join(LAB, 'summary.csv')}")


if __name__ == "__main__":
    main()
