"""
Bigger brains when needed: brain size as a heritable trait.

Every creature has room for brain_max hidden units but uses only the first nact of them.
Unused units have all-zero weights, so they contribute exactly nothing, and the compiled
kernel skips them (no time spent on them either).

  inheritance  a child has its parent's nact, plus or minus one with probability
               nact_mut_p (half each way), kept within [nact_min, brain_max];
  growth       a newly added unit starts with all-zero weights (disconnected); later
               mutations can wire it in;
  shrinking    the dropped unit's weights are zeroed;
  cost         each unit above nact_init costs unit_cost energy per tick and each unit
               below it saves the same, so a bigger brain only spreads if it pays for
               itself (and the founders' own brain costs nothing extra).

OPEN BRAINS (brain_open=True; Soros & Stanley 2014, condition 4: "the potential complexity of
the phenotype should not be limited by its representation"): there is no fixed brain_max.
The weight arrays start with room for nact_init + 8 units and are widened (new units all-zero)
whenever a child needs more, up to a safety ceiling (brain_ceiling, default 128 = 5x the
founders' brain; reaching it is logged in brains.csv as max_units).

Controls: brain_evolve=False keeps nact at nact_init. brain_wired=False lets nact change
exactly the same way and costs the same, but units beyond nact_init are never wired in (they
stay all-zero), so a bigger brain is pure cost: brain growth there is drift, not need.
brains.csv logs brain sizes every 1,000 ticks (and, in the several-ways-to-live world,
per gut group). Brain-size changes use their own random stream (rng_brain), saved in
checkpoints.

BrainMixin is combined with a world class: WaysBrainSim (several ways to live) and
PredBrainSim (arms race worlds).
"""
import csv
import os
import pickle

import numpy as np

from engine import MAX_AGE
from predators import PredSim
from ways import WaysSim


def _pad(a, axis, size):
    a = np.asarray(a)
    if a.shape[axis] >= size:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (0, size - a.shape[axis])
    return np.pad(a, pad)


class BrainMixin:
    _FIELDS_EXTRA = ["nact"]

    def __init__(self, *args, brain_evolve=True, brain_wired=True, brain_max=48, nact_init=24, nact_min=4,
                 nact_mut_p=0.05, unit_cost=0.0002, brain_open=False, brain_ceiling=128, **kw):
        if brain_open:
            brain_max = int(nact_init) + 8  # starting room only; widened on demand
        H = int(brain_max)
        f = kw.get("founders")
        if f is not None:  # pad the founders' brains with unused (all-zero) units
            f = dict(f)
            f["w1"], f["b1"] = _pad(f["w1"], 2, H), _pad(f["b1"], 1, H)
            f["w2"] = _pad(f["w2"], 1, H)
            f["wh"] = _pad(_pad(f["wh"], 1, H), 2, H)
            if f.get("ws") is not None:
                f["ws"] = _pad(f["ws"], 1, H)
            kw["founders"] = f
        kw["hidden"] = H
        super().__init__(*args, **kw)
        seed = int(kw.get("seed", args[0] if args else 1))
        self.rng_brain = np.random.default_rng([seed, 9191])
        self.cfg.update(brain_evolve=bool(brain_evolve), brain_wired=bool(brain_wired), brain_max=H, nact_init=int(nact_init), nact_min=int(nact_min),
                        nact_mut_p=float(nact_mut_p), unit_cost=float(unit_cost))
        if brain_open:
            self.cfg.update(brain_open=True, brain_ceiling=int(brain_ceiling))
        idx = np.flatnonzero(self.alive)
        self.nact[idx] = int(nact_init)
        self._zero_units(idx, self.nact[idx])

    def _alloc(self, cap):
        super()._alloc(cap)
        self.nact = np.zeros(cap, np.int64)

    def _zero_units(self, s, keep):
        """Zero every weight of units >= keep (per agent) in slots s."""
        if not len(s):
            return
        H = self.H
        m = np.arange(H)[None, :] >= np.asarray(keep)[:, None]  # n x H, True = unit unused
        self.w1[s] = self.w1[s] * ~m[:, None, :]
        self.b1[s] = self.b1[s] * ~m
        self.w2[s] = self.w2[s] * ~m[:, :, None]
        self.wh[s] = self.wh[s] * ~m[:, :, None] * ~m[:, None, :]
        self.ws[s] = self.ws[s] * ~m[:, :, None]
        self.hstate[s] = self.hstate[s] * ~m

    def _widen(self, newH):
        """Give every creature room for newH hidden units (the new units are all-zero: unused)."""
        H = self.H
        if newH <= H:
            return
        d = newH - H
        def pad(a, axes):
            p = [(0, 0)] * a.ndim
            for ax in axes:
                p[ax] = (0, d)
            return np.pad(a, p)
        self.w1 = pad(self.w1, [2]); self.b1 = pad(self.b1, [1]); self.w2 = pad(self.w2, [1])
        self.wh = pad(self.wh, [1, 2]); self.hstate = pad(self.hstate, [1]); self.ws = pad(self.ws, [1])
        self.H = newH
        self.cfg["hidden"] = newH
        self.cfg["brain_max"] = newH

    def _nact_args(self):
        return self.nact, True

    def _spawn_children(self, par):
        s = super()._spawn_children(par)
        c = self.cfg
        na = self.nact[par]
        if c["brain_evolve"]:
            u = self.rng_brain.random(len(par))
            p = c["nact_mut_p"]
            delta = np.where(u < p / 2, 1, np.where(u < p, -1, 0))
            top = c["brain_ceiling"] if c.get("brain_open") else c["brain_max"]
            child = np.clip(na + delta, c["nact_min"], top)
            if c.get("brain_open") and len(child) and child.max() > self.H:
                self._widen(min(c["brain_ceiling"], int(child.max()) + 8))
        else:
            child = na.copy()
        self.nact[s] = child
        # units the parent did not use stay zero (a grown unit starts disconnected);
        # a dropped unit is zeroed
        keep = np.minimum(na, child)
        if not c.get("brain_wired", True):
            keep = np.minimum(keep, c["nact_init"])  # control: units beyond the founders' never get wired
        self._zero_units(s, keep)
        return s

    def _after_act(self, idx, act, x0, y0, e, keys, dead):
        super()._after_act(idx, act, x0, y0, e, keys, dead)
        # cost relative to the founders' brain: each unit above nact_init costs unit_cost per
        # tick, each unit below it saves the same (so the founders pay nothing extra)
        e -= self.cfg["unit_cost"] * (self.nact[idx] - self.cfg["nact_init"])
        newly = (dead == 0) & ((e <= 0) | (self.age[idx] >= MAX_AGE))
        dead[newly] = 1

    def _novelty(self):
        super()._novelty()
        if not self.out_dir:
            return
        idx = np.flatnonzero(self.alive)
        na = self.nact[idx]
        path = os.path.join(self.out_dir, "brains.csv")
        new = not os.path.exists(path)
        H = self.cfg["brain_max"]
        bins = list(range(0, H + 1, 4))
        hist = np.histogram(na, bins=bins + [H + 1])[0] / max(1, len(idx))
        extra, names = [], []
        if hasattr(self, "gut"):
            g = self.gut[idx]
            for name, m in (("surface", g < 1 / 3), ("diggers", g > 2 / 3)):
                names.append(f"mean_units_{name}")
                extra.append(round(float(na[m].mean()), 3) if m.any() else "")
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "population", "mean_units", "min_units", "max_units"] + names
                           + [f"units_{b}-{b + 3}" for b in bins])
            w.writerow([self.tick, len(idx), round(float(na.mean()), 3) if len(idx) else 0,
                        int(na.min()) if len(idx) else 0, int(na.max()) if len(idx) else 0] + extra
                       + [round(float(h), 4) for h in hist])

    def _checkpoint_extra(self):
        d = super()._checkpoint_extra()
        d.update(rng_brain_state=self.rng_brain.bit_generator.state, brain_marker=True)
        return d

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.rng_brain = np.random.default_rng()
        sim.rng_brain.bit_generator.state = d["rng_brain_state"]
        return sim


class WaysBrainSim(BrainMixin, WaysSim):
    _FIELDS = WaysSim._FIELDS + BrainMixin._FIELDS_EXTRA


class PredBrainSim(BrainMixin, PredSim):
    _FIELDS = PredSim._FIELDS + BrainMixin._FIELDS_EXTRA
