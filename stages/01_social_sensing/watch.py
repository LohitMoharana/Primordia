"""
Watch one agent think and move, frame by frame, against the real food grid.
This is how you check whether evolution actually produced foraging behavior,
or whether a lucky lineage is just surviving by chance.

Usage:
    python3 watch.py runs/seed1/checkpoint_9500.pkl            # oldest/most-evolved
    python3 watch.py runs/seed1/checkpoint_9500.pkl --steps 150
    python3 watch.py runs/seed1/checkpoint_9500.pkl --agent-id 842
"""
import argparse
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["world"], data["agents"]


def pick_agent(agents, agent_id=None):
    if agent_id is not None:
        for a in agents:
            if a.id == agent_id:
                return a
        raise ValueError(f"No agent with id {agent_id} in this checkpoint")
    # default: the one with the deepest generation = most evolutionary history behind it
    return max(agents, key=lambda a: a.generation)


def run_watch(world, agent, steps, rng):
    """Step just this one agent forward against a frozen copy of the world,
    recording its trajectory, energy, and chosen actions."""
    w = copy.deepcopy(world)
    a = copy.deepcopy(agent)
    frames = []
    for t in range(steps):
        w.step()
        pre_energy = a.energy
        x0, y0 = a.x, a.y
        a.step(w, rng)  # single real step — no separate/duplicate sensing or action draw
        frames.append({
            "t": t, "x": x0, "y": y0, "action": a.last_action,
            "energy_before": pre_energy, "energy_after": a.energy,
            "food_grid": w.food.copy(), "alive": a.alive,
        })
        if not a.alive:
            break
    return frames


def render_gif(frames, world_size, out_path="agent_watch.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(i):
        ax.clear()
        f = frames[i]
        ax.imshow(f["food_grid"].T, cmap="Greens", vmin=0, vmax=1, origin="upper")
        ax.plot(f["x"], f["y"], "o", color="red", markersize=10)
        ax.set_title(f"t={f['t']}  action={f['action']}  energy={f['energy_after']:.2f}")
        ax.set_xlim(-0.5, world_size - 0.5)
        ax.set_ylim(world_size - 0.5, -0.5)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=150)
    ani.save(out_path, writer="pillow")
    print(f"Saved {out_path} ({len(frames)} frames)")


def summarize(frames):
    actions = [f["action"] for f in frames]
    from collections import Counter
    counts = Counter(actions)
    print("\nAction counts over this run:")
    for act, n in counts.most_common():
        print(f"  {act:10s} {n}")
    energies = [f["energy_after"] for f in frames]
    print(f"\nEnergy: start={frames[0]['energy_before']:.2f} "
          f"end={energies[-1]:.2f} min={min(energies):.2f} max={max(energies):.2f}")
    print(f"Survived {len(frames)} steps. Alive at end: {frames[-1]['alive']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--agent-id", type=int, default=None)
    args = parser.parse_args()

    world, agents = load_checkpoint(args.checkpoint)
    agent = pick_agent(agents, args.agent_id)
    print(f"Watching agent {agent.id}, generation {agent.generation}, "
          f"age {agent.age}, energy {agent.energy:.2f}")

    rng = np.random.default_rng(123)
    frames = run_watch(world, agent, args.steps, rng)
    summarize(frames)
    render_gif(frames, world.size)
