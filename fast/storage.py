"""
Agents that change the world, part 1: food storage.

Creatures can put energy into the ground of the cell they stand on and take it out
later. Stores stay in the world after their maker moves on or dies, so later
generations live in a world partly built by earlier ones.

  store     the "mark" action (unused in these worlds): moves up to STORE_AMOUNT energy
            from the creature into its cell's store (never below 1 energy left).
  retrieve  the "mate" action (unused): takes up to TAKE_AMOUNT from the cell's store;
            several takers in one cell are served in random order.
  spoil     each store loses `spoil` of its contents per tick.
  sensing   stores within 2 cells, in inputs 150-174 (the unused marker slot).

Stores are public: anyone standing on one can take from it. storage.csv logs how much is
stored, deposits, retrievals, and how much of what was taken came back to the creature
that put it there (the last depositor of the cell) versus someone else.

storage=False is the control: the same actions exist and cost the same, but do nothing.
"""
import csv
import os
import pickle

import numpy as np

from engine import A, MAX_AGE, MAX_ENERGY
from predators import PredSim

STORE_AMOUNT = 0.5
TAKE_AMOUNT = 0.5


class StoreSim(PredSim):
    _METER = PredSim._METER + ["_store_stats"]

    def __init__(self, *args, storage=True, spoil=0.001, store_cap=20.0, predators=False, **kw):
        super().__init__(*args, predators=predators, **kw)
        self.cfg.update(storage=bool(storage), spoil=float(spoil), store_cap=float(store_cap))
        S = self.size
        self.store = np.zeros((S, S))
        self.store_owner = np.full((S, S), -1, np.int64)  # id of the last creature to deposit here
        self._store_stats = self._new_stats()

    @staticmethod
    def _new_stats():
        return dict(deposits=0, deposited=0.0, takes=0, taken=0.0, taken_own=0.0, spoiled=0.0)

    # ---------------------------------------------------------------- hooks
    def _mark_slot_args(self):
        if self.cfg.get("storage", False):
            return self.store, True
        return super()._mark_slot_args()

    def _after_act(self, idx, act, x0, y0, e, keys, dead):
        super()._after_act(idx, act, x0, y0, e, keys, dead)
        c, S = self.cfg, self.size
        if not c.get("storage", False):
            return
        st = self._store_stats
        touched = []
        # deposits (start-of-tick cell, like eating); a creature keeps at least 1 energy
        dep = np.flatnonzero((act == A["mark"]) & (dead == 0))
        if len(dep):
            room = np.maximum(0.0, c["store_cap"] - self.store[x0[dep], y0[dep]])
            amt = np.minimum(np.minimum(STORE_AMOUNT, np.maximum(0.0, e[dep] - 1.0)), room)
            ok = amt > 0
            dep, amt = dep[ok], amt[ok]
            if len(dep):
                e[dep] -= amt
                np.add.at(self.store, (x0[dep], y0[dep]), amt)
                self.store_owner[x0[dep], y0[dep]] = self.aid[idx[dep]]
                st["deposits"] += len(dep)
                st["deposited"] += float(amt.sum())
                touched.append(dep)
        # retrievals: takers in one cell in random-key order, up to TAKE_AMOUNT each
        tk = np.flatnonzero((act == A["mate"]) & (dead == 0))
        if len(tk):
            cell = x0[tk] * S + y0[tk]
            order = np.lexsort((keys[tk], cell))
            cell_s = cell[order]
            first = np.r_[True, cell_s[1:] != cell_s[:-1]]
            start = np.flatnonzero(first)
            rank = np.arange(len(tk)) - np.repeat(start, np.diff(np.r_[start, len(tk)]))
            avail = self.store.ravel()[cell_s]
            got_s = np.clip(avail - TAKE_AMOUNT * rank, 0.0, TAKE_AMOUNT)
            got = np.empty(len(tk))
            got[order] = got_s
            got = np.minimum(got, MAX_ENERGY - e[tk])  # no taking more than fits
            got = np.maximum(got, 0.0)
            np.add.at(self.store.ravel(), cell, -got)
            np.maximum(self.store, 0.0, out=self.store)
            e[tk] += got
            own = self.store_owner.ravel()[cell] == self.aid[idx[tk]]
            st["takes"] += int((got > 0).sum())
            st["taken"] += float(got.sum())
            st["taken_own"] += float(got[own].sum())
            touched.append(tk)
        if touched:
            t = np.concatenate(touched)
            dead[t] = ((e[t] <= 0) | (self.age[idx[t]] >= MAX_AGE)).astype(dead.dtype)

    def _world_step(self, foods=None):
        super()._world_step(foods)
        if self.cfg.get("storage", False):
            sp = self.store * self.cfg["spoil"]
            self._store_stats["spoiled"] += float(sp.sum())
            self.store -= sp

    # ---------------------------------------------------------------- logging
    def _novelty(self):
        super()._novelty()
        if not self.out_dir:
            return
        st = self._store_stats
        path = os.path.join(self.out_dir, "storage.csv")
        new = not os.path.exists(path)
        nz = self.store > 0.01
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "population", "stored_total", "cells_with_store", "deposits", "deposited",
                            "takes", "taken", "taken_by_depositor", "spoiled"])
            w.writerow([self.tick, self.population, round(float(self.store.sum()), 2), int(nz.sum()),
                        st["deposits"], round(st["deposited"], 2), st["takes"], round(st["taken"], 2),
                        round(st["taken_own"], 2), round(st["spoiled"], 2)])
        self._store_stats = self._new_stats()

    # ---------------------------------------------------------------- save / load
    def _checkpoint_extra(self):
        d = super()._checkpoint_extra()
        d.update(store=self.store, store_owner=self.store_owner)
        return d

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.store = d["store"]
        sim.store_owner = d["store_owner"]
        if not hasattr(sim, "_store_stats"):
            sim._store_stats = cls._new_stats()
        return sim
