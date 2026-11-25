"""
Several ways to live: a second food type that takes a different skill, and a
heritable digestion trade-off.

  surface food  the usual drifting patches, eaten with "eat".
  buried food   grows slowly in a few fixed patches of rich soil; only digging gets it
                (the "mark" action, unused in these worlds, is the dig action). Worth more
                per bite (DIG_GAIN 1.0 vs EAT_GAIN 0.5). Prey sense buried food in inputs
                150-174 (the unused marker slot).
  gut           every creature carries a heritable number g in [0, 1] (mutates by
                N(0, gut_mut) at birth). Energy per unit eaten: surface (1-g)^q, buried b + (1-b) g^q (b = buried_base:
                even a surface specialist gets a little from digging, so digging can start).
                With q > 1 a generalist (g = 0.5) gets less than half of each, so being
                good at one food costs you on the other.

Only the trade-off is set; what creatures do, where they go and whether two kinds of
creature appear is left to evolution.

buried=False is the control: the same gut trade-off, but no buried food, so only surface
eating can pay.

Buried food, soil and gut use their own random stream (rng_ways), saved in checkpoints.
WaysSim derives from PredSim, so predators can be added later (predators=True).
"""
import csv
import os
import pickle

import numpy as np

from engine import A, MAX_AGE, MAX_ENERGY
from predators import PredSim

DIG_GAIN = 1.0


class WaysSim(PredSim):
    _FIELDS = PredSim._FIELDS + ["gut"]
    _METER = PredSim._METER + ["_dig_counts"]

    def __init__(self, *args, buried=True, n_soil=6, soil_sigma=6.0, buried_regrow=0.02, buried_max=2.0,
                 gut_q=1.5, gut_mut=0.05, gut_init_max=1.0, buried_base=0.0, predators=False, **kw):
        super().__init__(*args, predators=predators, **kw)
        seed = int(kw.get("seed", args[0] if args else 1))
        self.rng_ways = np.random.default_rng([seed, 4242])
        self.cfg.update(buried=bool(buried), n_soil=int(n_soil), soil_sigma=float(soil_sigma),
                        buried_regrow=float(buried_regrow), buried_max=float(buried_max),
                        gut_q=float(gut_q), gut_mut=float(gut_mut), gut_init_max=float(gut_init_max),
                        buried_base=float(buried_base))
        S = self.size
        rw = self.rng_ways
        # rich soil: n_soil fixed Gaussian patches (toroidal); buried food regrows only there
        g = np.arange(S)
        centres = rw.uniform(0, S, (int(n_soil), 2))

        def bump(c):
            d = np.abs(g[None, :] - c[:, None])
            d = np.minimum(d, S - d)
            return np.exp(-d ** 2 / (2 * soil_sigma ** 2))
        self.soil_map = np.minimum(np.einsum("kx,ky->xy", bump(centres[:, 0]), bump(centres[:, 1])), 1.0)
        self.soil_map[self.wall > 0] = 0.0
        self.buried = np.zeros((S, S))
        if buried:
            self.buried[rw.random((S, S)) < self.soil_map * 0.5] = buried_max
        # founders: gut drawn uniformly from [0, gut_init_max] (standing variation)
        idx = np.flatnonzero(self.alive)
        self.gut[idx] = rw.uniform(0, gut_init_max, len(idx))
        self._dig_counts = 0

    # ---------------------------------------------------------------- per-agent gut
    def _alloc(self, cap):
        super()._alloc(cap)
        self.gut = np.zeros(cap)

    def _spawn_children(self, par):
        s = super()._spawn_children(par)
        c = self.cfg
        self.gut[s] = np.clip(self.gut[par] + self.rng_ways.normal(0, c["gut_mut"], len(par)), 0.0, 1.0)
        return s

    # ---------------------------------------------------------------- hooks
    def _mark_slot_args(self):
        if self.cfg.get("buried", False):
            return self.buried, True
        return super()._mark_slot_args()

    def _dig_eff(self, i):
        b = self.cfg.get("buried_base", 0.0)
        return b + (1.0 - b) * self.gut[i] ** self.cfg["gut_q"]

    def _eat_eff(self, idx):
        return (1.0 - self.gut[idx]) ** self.cfg["gut_q"]

    def _after_act(self, idx, act, x0, y0, e, keys, dead):
        c, S = self.cfg, self.size
        if not c.get("buried", False):
            return
        dig = np.flatnonzero(act == A["mark"])  # dig action (its energy cost was paid in the action phase)
        self._dig_counts += len(dig)
        if not len(dig):
            return
        # diggers in one cell take buried food in random-key order, up to DIG_GAIN each (as eating)
        cell = x0[dig] * S + y0[dig]
        order = np.lexsort((keys[dig], cell))
        cell_s = cell[order]
        first = np.r_[True, cell_s[1:] != cell_s[:-1]]
        start = np.flatnonzero(first)
        rank = np.arange(len(dig)) - np.repeat(start, np.diff(np.r_[start, len(dig)]))
        avail = self.buried.ravel()[cell_s]
        gain = np.clip(avail - DIG_GAIN * rank, 0.0, DIG_GAIN)
        gg = np.empty(len(dig))
        gg[order] = gain
        np.add.at(self.buried.ravel(), cell, -gg)
        np.maximum(self.buried, 0.0, out=self.buried)
        e[dig] = np.minimum(e[dig] + gg * self._dig_eff(idx[dig]), MAX_ENERGY)
        # a digger saved by its meal survives (same death rule as the action phase)
        dead[dig] = ((e[dig] <= 0) | (self.age[idx[dig]] >= MAX_AGE)).astype(dead.dtype)

    def _world_step(self, foods=None):
        super()._world_step(foods)
        c = self.cfg
        if c.get("buried", False):
            rr = self.rng_ways.random((self.size, self.size))
            grow = (self.buried < 0.05) & (rr < self.soil_map * c["buried_regrow"])
            self.buried[grow] = c["buried_max"]

    # ---------------------------------------------------------------- logging
    def _novelty(self):
        cnt = getattr(self, "_act_counts", None)
        total = int(cnt.sum()) if cnt is not None else 0
        super()._novelty()
        if not self.out_dir:
            return
        idx = np.flatnonzero(self.alive)
        gut = self.gut[idx]
        hist = np.histogram(gut, bins=10, range=(0, 1))[0] / max(1, len(idx))
        path = os.path.join(self.out_dir, "ways.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "population", "mean_gut", "share_surface_specialists", "share_diggers",
                            "dig_share_of_actions", "buried_food"] + [f"gut_{b / 10:.1f}" for b in range(10)])
            w.writerow([self.tick, len(idx), round(float(gut.mean()) if len(idx) else 0, 4),
                        round(float((gut < 1 / 3).mean()) if len(idx) else 0, 4),
                        round(float((gut > 2 / 3).mean()) if len(idx) else 0, 4),
                        round(self._dig_counts / max(1, total), 4), round(float(self.buried.sum()), 1)]
                       + [round(float(h), 4) for h in hist])
        self._dig_counts = 0

    # ---------------------------------------------------------------- save / load
    def _checkpoint_extra(self):
        d = super()._checkpoint_extra()
        d.update(buried=self.buried, soil_map=self.soil_map, rng_ways_state=self.rng_ways.bit_generator.state)
        return d

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.buried = d["buried"]
        sim.soil_map = d["soil_map"]
        sim.rng_ways = np.random.default_rng()
        sim.rng_ways.bit_generator.state = d["rng_ways_state"]
        if not hasattr(sim, "_dig_counts"):
            sim._dig_counts = 0
        return sim
