"""
Controlled on/off comparison for a stage, at matched generation depth.

  python3 evaluate.py 02_communication

Picks the pair of checkpoints (one per arm) whose generation depths are
closest, samples 50 agents per arm, runs each for 150 steps, and reports
eat-fraction, energy change, survival, and how often each action is used
(so an unused channel shows up as near-zero usage). Appends a row to
eval_history.csv.
"""
import csv
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

LAB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LAB)  # bundle root; code_dir in stages.json is relative to it


def _code(stage):
    cd = stage["code_dir"]
    return cd if os.path.isabs(cd) else os.path.join(ROOT, cd)
STEPS, N, SEED = 150, 50, 7


def gen_by_tick(run_dir):
    out = {}
    with open(os.path.join(run_dir, "log.csv")) as f:
        for row in csv.DictReader(f):
            out[int(row["tick"])] = int(row["max_generation"])
    return out


def ckpts_with_gen(run_dir):
    gens = gen_by_tick(run_dir)
    res = []
    for name in os.listdir(run_dir):
        if name.startswith("checkpoint_") and name.endswith(".pkl"):
            t = int(name[len("checkpoint_"):-4])
            if t in gens:
                res.append((t, gens[t], os.path.join(run_dir, name)))
    return sorted(res)


def matched_pair(on_dir, off_dir):
    on, off = ckpts_with_gen(on_dir), ckpts_with_gen(off_dir)
    # Deepest pair whose generations are within TOL of each other; if none,
    # fall back to the closest-generation pair. (Preferring an exact match
    # first wrongly picked shallow pairs once old checkpoints were pruned.)
    TOL = 3
    pairs = [(a, b) for a in on for b in off]
    close = [p for p in pairs if abs(p[0][1] - p[1][1]) <= TOL]
    if close:
        return max(close, key=lambda p: (min(p[0][1], p[1][1]), -abs(p[0][1] - p[1][1])))
    return min(pairs, key=lambda p: abs(p[0][1] - p[1][1]))


def profile(ckpt_path, run_agent):
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    world, agents = data["world"], data["agents"]
    rng = np.random.default_rng(SEED)
    sample = rng.choice(agents, size=min(N, len(agents)), replace=False)
    eat, energy, surv, actions = [], [], [], Counter()
    for a in sample:
        r = run_agent(world, a, STEPS, np.random.default_rng(int(a.id) + 1))
        eat.append(r["eat_frac"])
        energy.append(r["end_energy"] - r["start_energy"])
        surv.append(r["survived_steps"])
        actions.update(r["actions"])
    total = sum(actions.values()) or 1
    return {
        "population": len(agents),
        "gen": max(a.generation for a in agents),
        "eat": float(np.mean(eat)),
        "energy": float(np.mean(energy)),
        "survival": float(np.mean(surv)),
        "improving": float(np.mean(np.array(energy) > 0)),
        "action_share": {k: v / total for k, v in sorted(actions.items())},
        # mean |recurrent weight| over the WHOLE population, if brains have memory
        "memory": (float(np.mean([a.brain.memory_strength() for a in agents]))
                   if hasattr(agents[0].brain, "memory_strength") else None),
    }


def main(stage_name, seed=1):
    with open(os.path.join(LAB, "stages.json")) as f:
        stage = next(s for s in json.load(f)["stages"] if s["name"] == stage_name)
    code = _code(stage)
    sys.path.insert(0, code)  # unpickle with THIS stage's classes
    os.chdir(code)
    from compare import run_agent

    dirs = {r["arm"]: os.path.join(code, r["dir"].replace("seed1", f"seed{seed}")) for r in stage["runs"]}
    (t_on, g_on, p_on), (t_off, g_off, p_off) = matched_pair(dirs["on"], dirs["off"])
    on, off = profile(p_on, run_agent), profile(p_off, run_agent)

    print(f"Stage {stage_name} (seed {seed}): ON tick {t_on} (gen {g_on}) vs OFF tick {t_off} (gen {g_off})")
    print(f"{'metric':<22}{'OFF':>10}{'ON':>10}{'ON-OFF':>10}")
    for k, label in [("eat", "eat-fraction"), ("energy", "energy change"),
                     ("survival", "steps survived /150"), ("improving", "share improving"),
                     ("population", "population")]:
        print(f"{label:<22}{off[k]:>10.3f}{on[k]:>10.3f}{on[k]-off[k]:>+10.3f}")
    if on["memory"] is not None:
        print(f"{'memory strength |wh|':<22}{off['memory']:>10.4f}{on['memory']:>10.4f}"
              f"{on['memory']-off['memory']:>+10.4f}")
    print("\nAction share (ON):  " + ", ".join(f"{k} {v:.1%}" for k, v in on["action_share"].items()))
    print("Action share (OFF): " + ", ".join(f"{k} {v:.1%}" for k, v in off["action_share"].items()))

    hist = os.path.join(LAB, "eval_history.csv")
    new = not os.path.exists(hist)
    with open(hist, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["stage", "gen_on", "gen_off", "eat_diff", "energy_diff",
                        "survival_diff", "pop_on", "pop_off"])
        w.writerow([stage_name, g_on, g_off, round(on["eat"] - off["eat"], 3),
                    round(on["energy"] - off["energy"], 3),
                    round(on["survival"] - off["survival"], 2), on["population"], off["population"]])


if __name__ == "__main__":
    # usage: python3 lab/evaluate.py <stage> [seed]
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
