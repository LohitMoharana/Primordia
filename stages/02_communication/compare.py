"""
Two honesty checks on whether evolution is really doing anything:

1. EARLY vs LATE: pull an agent from an early checkpoint (mostly gen 0-5,
   still close to random) and one from a late checkpoint, run both through
   the same identical food layout, and compare how they do. If evolution
   isn't doing anything, these should look the same.

2. POPULATION SPREAD: instead of just the single best/deepest-generation
   agent, sample many agents from one late checkpoint and compare their
   action profiles. If "eat constantly" is a universal, convergent
   strategy, most of the population should show it — not just one lineage
   that got lucky.
"""
import argparse
import copy
import pickle
from collections import Counter

import numpy as np


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["world"], data["agents"]


def run_agent(world, agent, steps, rng):
    w = copy.deepcopy(world)
    a = copy.deepcopy(agent)
    actions = []
    start_energy = a.energy
    for _ in range(steps):
        w.step()
        a.step(w, rng)
        actions.append(a.last_action)
        if not a.alive:
            break
    return {
        "survived_steps": len(actions),
        "alive": a.alive,
        "start_energy": start_energy,
        "end_energy": a.energy,
        "eat_frac": actions.count("eat") / len(actions) if actions else 0,
        "actions": Counter(actions),
    }


def early_vs_late(early_ckpt, late_ckpt, steps, seed):
    early_world, early_agents = load_checkpoint(early_ckpt)
    late_world, late_agents = load_checkpoint(late_ckpt)

    # use the LATE world's food layout for both, so it's a fair identical test
    test_world = late_world

    early_agent = min(early_agents, key=lambda a: a.generation)
    late_agent = max(late_agents, key=lambda a: a.generation)

    print(f"Early agent: gen {early_agent.generation}, from {early_ckpt}")
    print(f"Late  agent: gen {late_agent.generation}, from {late_ckpt}")
    print(f"(Both dropped into the SAME food layout, for fairness)\n")

    for label, agent in [("EARLY", early_agent), ("LATE", late_agent)]:
        rng = np.random.default_rng(seed)  # identical random seed for both
        result = run_agent(test_world, agent, steps, rng)
        print(f"--- {label} (gen {agent.generation}) ---")
        print(f"  survived {result['survived_steps']}/{steps} steps, alive={result['alive']}")
        print(f"  energy: {result['start_energy']:.2f} -> {result['end_energy']:.2f}")
        print(f"  ate on {result['eat_frac']*100:.1f}% of steps")
        print(f"  actions: {dict(result['actions'])}\n")


def population_spread(checkpoint, steps, n_sample, seed):
    world, agents = load_checkpoint(checkpoint)
    rng_pick = np.random.default_rng(seed)
    sample = rng_pick.choice(agents, size=min(n_sample, len(agents)), replace=False)

    print(f"Sampling {len(sample)} agents from {checkpoint} "
          f"(population at checkpoint: {len(agents)})\n")

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

    print(f"Eat-fraction across population: mean={eat_fracs.mean():.2f} "
          f"std={eat_fracs.std():.2f} min={eat_fracs.min():.2f} max={eat_fracs.max():.2f}")
    print(f"Energy change:  mean={energy_deltas.mean():+.2f} std={energy_deltas.std():.2f}")
    print(f"Steps survived: mean={survived.mean():.1f} / {steps}")
    print(f"\nFraction of sampled agents with positive energy change: "
          f"{(energy_deltas > 0).mean()*100:.1f}%")
    print("If most of the population (not just one lineage) eats often and gains")
    print("energy, that's evidence of a convergent, population-wide strategy —")
    print("not one lucky individual.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("early-vs-late")
    p1.add_argument("early_checkpoint")
    p1.add_argument("late_checkpoint")
    p1.add_argument("--steps", type=int, default=150)
    p1.add_argument("--seed", type=int, default=42)

    p2 = sub.add_parser("population")
    p2.add_argument("checkpoint")
    p2.add_argument("--steps", type=int, default=150)
    p2.add_argument("--n", type=int, default=40)
    p2.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    if args.mode == "early-vs-late":
        early_vs_late(args.early_checkpoint, args.late_checkpoint, args.steps, args.seed)
    else:
        population_spread(args.checkpoint, args.steps, args.n, args.seed)
