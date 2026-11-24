"""
The simulation loop. Spawns an initial population with RANDOM brains and
just... runs the world. There is no scripted survival strategy anywhere
in this file. If agents learn to find food, flee competition, or time
reproduction well, that behavior exists only because it survived.
"""
import csv
import os
import pickle
import numpy as np

from world import World
from agent import Agent

LOG_EVERY = 10
CHECKPOINT_EVERY = 500


class Simulation:
    def __init__(self, world_size=48, init_population=80, seed=0, out_dir="runs/seed0",
                 social_sensing=True, communication=True, seasons=False, memory=False):
        self.rng = np.random.default_rng(seed)
        self.world = World(size=world_size, seed=seed, social_sensing=social_sensing,
                            communication=communication, seasons=seasons, memory=memory)
        self.agents = []
        for _ in range(init_population):
            x = self.rng.integers(0, world_size)
            y = self.rng.integers(0, world_size)
            self.agents.append(Agent(x, y, rng=self.rng))
        self.tick = 0
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "log.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "tick", "population", "mean_energy", "mean_age",
                "max_generation", "mean_complexity", "total_food"
            ])

    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir, rng_seed=None):
        """Resume a run from a .pkl checkpoint — same world, same agents,
        same evolutionary history, continuing from that exact tick.

        out_dir should contain the checkpoint's log.csv (copy it alongside
        the checkpoint when moving machines) so the log continues in place
        rather than restarting from tick 0.
        """
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)

        sim = cls.__new__(cls)  # skip __init__'s fresh-world setup entirely
        sim.world = data["world"]
        sim.agents = data["agents"]
        sim.tick = data["tick"]
        # Agent ids come from a class counter that restarts at 0 in a new
        # process; continue it past the loaded agents so ids stay unique
        # (and max id + 1 keeps meaning "agents ever born in this run").
        Agent._next_id = max([a.id for a in sim.agents], default=-1) + 1
        if "next_id" in data:
            Agent._next_id = max(Agent._next_id, data["next_id"])

        # Migration: this checkpoint predates the social_sensing ablation flag.
        # Every run before that flag existed was unconditionally using real
        # occupancy data, so defaulting to True here restores exactly what
        # this run was already doing — it does not change its behavior.
        if not hasattr(sim.world, "social_sensing"):
            sim.world.social_sensing = True
            print("NOTE: checkpoint predates the social_sensing flag — "
                  "defaulted to True, matching this run's actual prior behavior.")
        if "rng_state" in data and rng_seed is None:
            # bit-exact continuation — this picks up EXACTLY where the original run was,
            # not a fresh random stream that happens to start from the same population.
            sim.rng = np.random.default_rng()
            sim.rng.bit_generator.state = data["rng_state"]
        else:
            # older checkpoint with no saved RNG state (or caller forced a seed):
            # this is NOT bit-exact. If this checkpoint's tick range was already logged
            # by a run that later died, re-running it now will NOT reproduce those exact
            # rows — expect small divergence in any already-logged, overlapping tick range.
            sim.rng = np.random.default_rng(rng_seed if rng_seed is not None else sim.tick)
            print("NOTE: no saved RNG state in this checkpoint — resuming with a fresh "
                  "random stream, not a bit-exact continuation. Any tick range already "
                  "logged before this checkpoint's save point may now get slightly "
                  "different values if re-run. Use clean_log.py before analyzing.")
        sim.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        sim.log_path = os.path.join(out_dir, "log.csv")

        if not os.path.exists(sim.log_path):
            # no prior log carried over — start a fresh one, but note where we resumed
            with open(sim.log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "tick", "population", "mean_energy", "mean_age",
                    "max_generation", "mean_complexity", "total_food"
                ])
            print(f"Note: no log.csv found in {out_dir} — starting a new one "
                  f"from tick {sim.tick}. Copy the original log.csv alongside the "
                  f"checkpoint next time to keep one continuous history.")
        else:
            print(f"Resuming from tick {sim.tick}, appending to existing {sim.log_path}")

        return sim

    def step(self):
        self.world.step()
        self.world.set_occupancy(self.agents)
        self.world.set_signals(self.agents)
        self.rng.shuffle(self.agents)  # no fixed turn order / no implicit hierarchy
        new_agents = []
        for a in self.agents:
            if not a.alive:
                continue
            child = a.step(self.world, self.rng)
            if child is not None:
                new_agents.append(child)
        self.agents = [a for a in self.agents if a.alive] + new_agents
        self.tick += 1

        if self.tick % LOG_EVERY == 0:
            self._log()
        if self.tick % CHECKPOINT_EVERY == 0:
            self._checkpoint()

    def _log(self):
        n = len(self.agents)
        if n == 0:
            row = [self.tick, 0, 0, 0, 0, 0, float(self.world.food.sum())]
        else:
            energies = [a.energy for a in self.agents]
            ages = [a.age for a in self.agents]
            gens = [a.generation for a in self.agents]
            complexities = [a.brain.complexity() for a in self.agents]
            row = [
                self.tick, n, np.mean(energies), np.mean(ages),
                max(gens), np.mean(complexities), float(self.world.food.sum())
            ]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _checkpoint(self):
        path = os.path.join(self.out_dir, f"checkpoint_{self.tick}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "tick": self.tick, "agents": self.agents, "world": self.world,
                "rng_state": self.rng.bit_generator.state,  # so a resume is bit-exact, not just "close"
                "next_id": Agent._next_id,  # total agents ever born, survives restarts
            }, f)

    def run(self, max_ticks=20000, extinction_stops=True):
        for _ in range(max_ticks):
            self.step()
            if extinction_stops and len(self.agents) == 0:
                print(f"Extinct at tick {self.tick}")
                break
        self._checkpoint()
        print(f"Done. Final population: {len(self.agents)} at tick {self.tick}")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sim = Simulation(seed=seed, out_dir=f"runs/seed{seed}")
    sim.run(max_ticks=20000)
