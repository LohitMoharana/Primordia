"""
Agents that change the world, part 2: building shelters.

The arms-race world (evolving predators, slow newcomers, 2 energy per catch), but no
shelters are placed. Instead creatures can build them:

  build    the "mark" action (unused in these worlds) turns the creature's own cell into
           a shelter, for BUILD_COST energy (only if it has more than BUILD_COST + 1).
           A shelter is what round 2 placed by hand: predators cannot catch prey standing
           in it, no food grows there, and prey sense shelters in inputs 100-124.
  decay    each shelter crumbles back to open ground with probability 1/shelter_life per
           tick, so a shelter can outlive its builder and serve later generations.

build=False is the control: the build action exists and costs the same, but does nothing.
building.csv logs shelters standing, built and crumbled, and how many prey stand in one.
Shelter decay uses its own random stream (rng_build), saved in checkpoints.
"""
import csv
import os
import pickle

import numpy as np

from engine import A, MAX_AGE
from predators import PredSim

BUILD_COST = 1.5


class BuildSim(PredSim):
    _METER = PredSim._METER + ["_build_stats"]

    def __init__(self, *args, build=True, shelter_life=1000.0, **kw):
        super().__init__(*args, **kw)
        seed = int(kw.get("seed", args[0] if args else 1))
        self.cfg.update(build=bool(build), shelter_life=float(shelter_life))
        self.rng_build = np.random.default_rng([seed, 8181])
        self._build_stats = dict(built=0, crumbled=0)

    def _after_act(self, idx, act, x0, y0, e, keys, dead):
        super()._after_act(idx, act, x0, y0, e, keys, dead)
        if not self.cfg.get("build", False):
            return
        b = np.flatnonzero((act == A["mark"]) & (dead == 0) & (e > BUILD_COST + 1.0))
        if not len(b):
            return
        bx, by = x0[b], y0[b]
        new = self.shelter[bx, by] == 0
        if not new.any():
            return
        b, bx, by = b[new], bx[new], by[new]
        # several builders in one new cell: only the first (in idx order) pays
        cell = bx * self.size + by
        _, first = np.unique(cell, return_index=True)
        b, bx, by = b[first], bx[first], by[first]
        e[b] -= BUILD_COST
        self.shelter[bx, by] = 1.0
        self.food[bx, by] = 0.0
        self._has_shelter = True
        self._build_stats["built"] += len(b)
        dead[b] = ((e[b] <= 0) | (self.age[idx[b]] >= MAX_AGE)).astype(dead.dtype)

    def _world_step(self, foods=None):
        if self.cfg.get("build", False) and self._has_shelter:
            on = np.flatnonzero(self.shelter.ravel() > 0)
            gone = on[self.rng_build.random(len(on)) < 1.0 / self.cfg["shelter_life"]]
            if len(gone):
                self.shelter.ravel()[gone] = 0.0
                self._build_stats["crumbled"] += len(gone)
                self._has_shelter = bool(self.shelter.any())
        super()._world_step(foods)

    def _novelty(self):
        super()._novelty()
        if not self.out_dir:
            return
        idx = np.flatnonzero(self.alive)
        inside = int((self.shelter[self.x[idx], self.y[idx]] > 0).sum()) if len(idx) else 0
        path = os.path.join(self.out_dir, "building.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "prey", "predators", "shelters", "built", "crumbled", "prey_in_shelter"])
            w.writerow([self.tick, len(idx), self.n_pred, int((self.shelter > 0).sum()),
                        self._build_stats["built"], self._build_stats["crumbled"], inside])
        self._build_stats = dict(built=0, crumbled=0)

    def _checkpoint_extra(self):
        d = super()._checkpoint_extra()
        d.update(rng_build_state=self.rng_build.bit_generator.state, build_marker=True)
        return d

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.rng_build = np.random.default_rng()
        sim.rng_build.bit_generator.state = d["rng_build_state"]
        if not hasattr(sim, "_build_stats"):
            sim._build_stats = dict(built=0, crumbled=0)
        return sim
