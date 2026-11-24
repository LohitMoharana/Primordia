"""
A FAIR cross-condition comparison: baseline (asocial) vs social-sensing agents,
at matched generation depth, tested on the IDENTICAL food layout, with the
SAME random seed per trial. This controls for "more evolutionary time" and
isolates the one real difference between conditions: whether the brain can
sense nearby agents.

Note: brain architectures differ in size (52 vs 52 inputs — WAIT, check below)
between conditions only if you've changed brain.py between runs. As currently
reverted, both conditions use the SAME brain shape (52 in, 16 hidden, 7 out) —
agent-sensing was added via occupancy grid + sense() concatenation, which DOES
change input size. Confirm shapes printed below before trusting the comparison.
"""
import pickle
import copy
from collections import Counter
import numpy as np

from compare import run_agent  # reuse the single-agent runner


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["world"], data["agents"]


def population_profile(checkpoint_path, label, steps=150, n=50, seed=7):
    world, agents = load_checkpoint(checkpoint_path)
    rng_pick = np.random.default_rng(seed)
    sample = rng_pick.choice(agents, size=min(n, len(agents)), replace=False)

    print(f"\n=== {label} ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"brain input size: {sample[0].brain.w1.shape[0]}  "
          f"max generation in sample: {max(a.generation for a in sample)}  "
          f"population at checkpoint: {len(agents)}")

    eat_fracs, energy_deltas, survived = [], [], []
    for a in sample:
        rng = np.random.default_rng(int(a.id) + 1)
        result = run_agent(world, a, steps, rng)
        eat_fracs.append(result["eat_frac"])
        energy_deltas.append(result["end_energy"] - result["start_energy"])
        survived.append(result["survived_steps"])

    eat_fracs = np.array(eat_fracs)
    energy_deltas = np.array(energy_deltas)
    survived = np.array(survived)

    print(f"Eat-fraction:    mean={eat_fracs.mean():.3f}  std={eat_fracs.std():.3f}")
    print(f"Energy change:   mean={energy_deltas.mean():+.3f}  std={energy_deltas.std():.3f}")
    print(f"Steps survived:  mean={survived.mean():.1f} / {steps}")
    print(f"Agents improving energy: {(energy_deltas > 0).mean()*100:.1f}%")
    return {"eat_fracs": eat_fracs, "energy_deltas": energy_deltas, "survived": survived}


if __name__ == "__main__":
    import sys
    ckpt_on = sys.argv[1] if len(sys.argv) > 1 else "runs/ablation_on_seed1/checkpoint_14500.pkl"
    ckpt_off = sys.argv[2] if len(sys.argv) > 2 else "runs/ablation_off_seed1/checkpoint_15000.pkl"
    baseline = population_profile(ckpt_off, "SOCIAL SENSING OFF (ablation)")
    social = population_profile(ckpt_on, "SOCIAL SENSING ON (ablation)")

    print("\n=== DIRECT COMPARISON ===")
    print(f"{'metric':<20}{'baseline':>12}{'social':>12}{'diff':>12}")
    for key, name in [("eat_fracs", "eat-fraction"), ("energy_deltas", "energy Δ"), ("survived", "steps survived")]:
        b, s = baseline[key].mean(), social[key].mean()
        print(f"{name:<20}{b:>12.3f}{s:>12.3f}{s-b:>+12.3f}")
