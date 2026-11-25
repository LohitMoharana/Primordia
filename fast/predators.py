"""
Evolving predators: a second species living in the same world as the prey.

Predators have their own small brains (52 inputs -> 16 hidden -> 7 actions) and
evolve the same way prey do: they gain energy by catching prey, pay for every
action, reproduce with mutation, and die of hunger or old age. Nothing about
hunting or fleeing is scripted.

  predator senses (5x5 around it): how many prey are in each cell, whether
  another predator is there, plus its own energy and age.
  predator actions: stay, N, S, E, W, attack, reproduce.
  attack: picks a random prey within radius 1 (3x3); catches it with
  probability CATCH_P. The prey dies; the predator gains KILL_GAIN.

Prey sense predators through the input slot reserved for them (inputs 125-149):
1 where a predator stood at the start of the tick, 0 elsewhere.

evolve=False is the control: predators reproduce as exact copies (no mutation),
so they can never get better at hunting; only the prey can change.

A floor keeps predators from vanishing for good: when fewer than `floor` are
alive, newcomers are added from the founder pool (the original random brains),
in both arms.
"""
import csv
import os
import pickle

import numpy as np

from engine import FastSim, MUT_RATE, MUT_STRENGTH
from kernels import HAVE_NUMBA
if HAVE_NUMBA:
    from kernels import pred_sense_think, pred_attacks

P_ACTIONS = ["stay", "N", "S", "E", "W", "attack", "reproduce"]
PA = {a: i for i, a in enumerate(P_ACTIONS)}
P_IN, P_HID, P_OUT = 52, 16, len(P_ACTIONS)


class PredSim(FastSim):
    def __init__(self, *args, predators=True, n_pred=60, pred_evolve=True, pred_floor=10,
                 pred_idle=0.02, pred_move=0.04, pred_attack=0.05, kill_gain=3.0, catch_p=0.5,
                 pred_max_e=15.0, pred_start_e=8.0, pred_rep_min=10.0, pred_rep_cost=6.0,
                 pred_max_age=800, pred_delay=0, pred_immigrants="founders", bank_size=30, **kw):
        super().__init__(*args, **kw)
        self.cfg.update(predators=bool(predators), n_pred=int(n_pred), pred_evolve=bool(pred_evolve),
                        pred_floor=int(pred_floor), pred_idle=pred_idle, pred_move=pred_move,
                        pred_attack=pred_attack, kill_gain=kill_gain, catch_p=catch_p,
                        pred_max_e=pred_max_e, pred_start_e=pred_start_e, pred_rep_min=pred_rep_min,
                        pred_rep_cost=pred_rep_cost, pred_max_age=int(pred_max_age),
                        pred_delay=int(pred_delay), pred_immigrants=str(pred_immigrants), bank_size=int(bank_size))
        self.pred_bank = None  # brains of the most recently dead predators (pred_immigrants="bank")
        r, S = self.rng, self.size
        n = int(n_pred)
        # founder pool: the original random predator brains (reused by the floor)
        self.pred_pool = dict(w1=r.normal(0, np.sqrt(2.0 / P_IN), (n, P_IN, P_HID)).astype(np.float32),
                              b1=np.zeros((n, P_HID), np.float32),
                              w2=r.normal(0, np.sqrt(2.0 / P_HID), (n, P_HID, P_OUT)).astype(np.float32),
                              b2=np.zeros((n, P_OUT), np.float32))
        self.pred = dict(x=np.zeros(0, np.int64), y=np.zeros(0, np.int64), e=np.zeros(0), age=np.zeros(0, np.int64),
                         gen=np.zeros(0, np.int64), **{k: v[:0] for k, v in self.pred_pool.items()})
        if predators and pred_delay == 0:
            self._add_pred(np.arange(n), r.integers(0, S, n), r.integers(0, S, n))
        self.pred_stats = dict(kills=0, attacks=0, births=0, deaths=0, immigrants=0)
        self._pred_act_counts = np.zeros(P_OUT, np.int64)
        self._prev_pred_share = None

    # ---------------------------------------------------------------- pool helpers
    def _add_pred(self, pool_idx, x, y, brains=None, energy=None, gen=None):
        b = brains if brains is not None else {k: v[pool_idx] for k, v in self.pred_pool.items()}
        n = len(x)
        P = self.pred
        P["x"] = np.r_[P["x"], np.asarray(x, np.int64) % self.size]
        P["y"] = np.r_[P["y"], np.asarray(y, np.int64) % self.size]
        P["e"] = np.r_[P["e"], np.full(n, self.cfg["pred_start_e"]) if energy is None else energy]
        P["age"] = np.r_[P["age"], np.zeros(n, np.int64)]
        P["gen"] = np.r_[P["gen"], np.zeros(n, np.int64) if gen is None else gen]
        for k in ("w1", "b1", "w2", "b2"):
            P[k] = np.concatenate([P[k], b[k]])

    def _keep_pred(self, keep):
        for k in self.pred:
            self.pred[k] = self.pred[k][keep]

    @property
    def n_pred(self):
        return len(self.pred["x"])

    # ---------------------------------------------------------------- tick
    def step(self):
        # prey sense where predators stand at the start of the tick
        if self.cfg.get("predators", False) and self.n_pred:
            g = np.zeros((self.size, self.size))
            g[self.pred["x"], self.pred["y"]] = 1.0
            self.pred_sense = g
        else:
            self.pred_sense = None
        super().step()
        if self.cfg.get("predators", False):
            self._pred_step()

    def _pred_step(self):
        c, r, S = self.cfg, self.rng, self.size
        if self.tick < c.get("pred_delay", 0):
            return  # prey settle first; predators arrive at pred_delay
        if self.tick == c.get("pred_delay", 0) and c.get("pred_delay", 0) > 0:
            n = c["n_pred"]
            self._add_pred(np.arange(n), r.integers(0, S, n), r.integers(0, S, n))
        P = self.pred
        if self.n_pred < c["pred_floor"]:
            k = c["pred_floor"] - self.n_pred
            bank = self.pred_bank
            if c.get("pred_immigrants", "founders") == "bank" and bank is not None and len(bank["w1"]):
                # newcomers are exact copies of recently dead predators, so a crash does not
                # throw away what predator evolution has found so far
                pick = r.integers(0, len(bank["w1"]), k)
                self._add_pred(None, r.integers(0, S, k), r.integers(0, S, k),
                               brains={q: bank[q][pick].copy() for q in ("w1", "b1", "w2", "b2")},
                               gen=bank["gen"][pick].copy())
            else:  # random brains from the founder pool (round 1 rule)
                self._add_pred(r.integers(0, len(self.pred_pool["w1"]), k), r.integers(0, S, k), r.integers(0, S, k))
            self.pred_stats["immigrants"] += k
        n = self.n_pred
        idx = np.flatnonzero(self.alive)
        prey_occ = np.bincount(self.x[idx] * S + self.y[idx], minlength=S * S).astype(np.float64).reshape(S, S)
        pred_occ = np.bincount(P["x"] * S + P["y"], minlength=S * S).astype(np.float64).reshape(S, S)
        u = r.random(n)
        if self._use_pred_kernel():
            act = np.empty(n, np.int64)
            pred_sense_think(P["x"], P["y"], P["e"].astype(np.float64), P["age"].astype(np.float64),
                             P["w1"], P["b1"], P["w2"], P["b2"], prey_occ, pred_occ, u,
                             float(c["pred_max_e"]), float(c["pred_max_age"]), act)
        else:
            act = self._pred_brain_py(prey_occ, pred_occ, u)
        self._pred_act_counts += np.bincount(act, minlength=P_OUT)
        P["age"] += 1
        x0, y0 = P["x"].copy(), P["y"].copy()
        e = P["e"]
        e[(act == PA["stay"]) | (act == PA["reproduce"])] -= c["pred_idle"]
        mv = (act >= PA["N"]) & (act <= PA["W"])
        P["x"] = np.where(act == PA["E"], x0 + 1, np.where(act == PA["W"], x0 - 1, x0)) % S
        P["y"] = np.where(act == PA["S"], y0 + 1, np.where(act == PA["N"], y0 - 1, y0)) % S
        e[mv] -= c["pred_move"]
        # attacks, in random order: a random prey within radius 1 (start-of-tick predator position)
        att = np.flatnonzero(act == PA["attack"])
        e[att] -= c["pred_attack"]
        if len(att) and len(idx):
            cell = self.x[idx] * S + self.y[idx]
            order = np.argsort(cell, kind="stable")
            cs = cell[order]
            ua, uc = r.random(len(att)), r.random(len(att))
            perm = r.permutation(len(att))
            if self._use_pred_kernel():
                st = np.zeros(2, np.int64)
                pred_attacks(att, perm, x0, y0, idx, order, cs, self.alive, ua, uc, float(c["catch_p"]),
                             float(c["kill_gain"]), e, S, st, self.shelter.ravel() > 0)
                self.pred_stats["attacks"] += int(st[0])
                self.pred_stats["kills"] += int(st[1])
            else:
                for q in perm:
                    j = att[q]
                    self.pred_stats["attacks"] += 1
                    cands = []
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            cc = ((x0[j] + dx) % S) * S + (y0[j] + dy) % S
                            if self._has_shelter and self.shelter.ravel()[cc] > 0:
                                continue  # prey in a shelter cannot be caught
                            lo, hi = np.searchsorted(cs, cc), np.searchsorted(cs, cc, side="right")
                            for s_ in order[lo:hi]:
                                if self.alive[idx[s_]]:
                                    cands.append(idx[s_])
                    if cands and ua[q] < c["catch_p"]:
                        victim = cands[int(uc[q] * len(cands))]
                        self.alive[victim] = False
                        e[j] += c["kill_gain"]
                        self.pred_stats["kills"] += 1
        np.minimum(e, c["pred_max_e"], out=e)
        # reproduction
        rep = np.flatnonzero((act == PA["reproduce"]) & (e >= c["pred_rep_min"]))
        if len(rep):
            e[rep] -= c["pred_rep_cost"]
            sd = MUT_STRENGTH if c["pred_evolve"] else 0.0

            def mut(a):  # a is a fresh copy already (fancy indexing)
                if sd > 0:
                    m = r.random(a.shape) < MUT_RATE
                    a[m] += r.normal(0, sd, int(m.sum())).astype(a.dtype)
                return a
            kids = {k: mut(P[k][rep]) for k in ("w1", "b1", "w2", "b2")}
            self._add_pred(None, P["x"][rep], P["y"][rep], brains=kids,
                           energy=np.full(len(rep), c["pred_rep_cost"] * 0.5), gen=P["gen"][rep] + 1)
            self.pred_stats["births"] += len(rep)
        # deaths
        dead = (P["e"] <= 0) | (P["age"] >= c["pred_max_age"])
        if dead.any():
            self.pred_stats["deaths"] += int(dead.sum())
            if c.get("pred_immigrants", "founders") == "bank":
                d = np.flatnonzero(dead)
                new = {q: P[q][d] for q in ("w1", "b1", "w2", "b2", "gen")}
                old = self.pred_bank
                keep = c.get("bank_size", 30)
                self.pred_bank = new if old is None else {q: np.concatenate([old[q], new[q]])[-keep:] for q in new}
            self._keep_pred(~dead)

    def _use_pred_kernel(self):
        return HAVE_NUMBA and self.cfg.get("pred_kernel", "numba") == "numba"

    def _pred_brain_py(self, prey_occ, pred_occ, u):
        """Reference (NumPy) predator brain: same inputs and sampling rule as pred_sense_think."""
        c, S, P = self.cfg, self.size, self.pred
        n = self.n_pred
        cx, cy = (P["x"][:, None] + self._ox) % S, (P["y"][:, None] + self._oy) % S
        X = np.zeros((n, P_IN), np.float32)
        X[:, :25] = np.log1p(prey_occ[cx, cy])
        po = pred_occ[cx, cy]
        po[:, 12] -= 1.0
        X[:, 25:50] = po > 0
        X[:, 50] = P["e"] / c["pred_max_e"]
        X[:, 51] = P["age"] / c["pred_max_age"]
        h = np.tanh(np.einsum("ni,nih->nh", X, P["w1"]) + P["b1"])
        out = np.einsum("nh,nho->no", h, P["w2"]) + P["b2"]
        out = (out - out.max(1, keepdims=True)).astype(np.float64)
        p = np.exp(out); p /= p.sum(1, keepdims=True)
        return np.minimum((np.cumsum(p, 1) < u[:, None]).sum(1), P_OUT - 1)

    # ---------------------------------------------------------------- logging
    def _novelty(self):
        super()._novelty()
        if not self.cfg.get("predators", False):
            return
        cnt = self._pred_act_counts
        share = cnt / max(1, cnt.sum())
        prev = self._prev_pred_share
        change = "" if prev is None else round(float(np.abs(share - prev).sum()), 4)
        path = os.path.join(self.out_dir, "predators.csv")
        new = not os.path.exists(path)
        P = self.pred
        spread = float(P["w2"].reshape(len(P["w2"]), -1).std(0).mean()) if self.n_pred > 1 else 0.0
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "predators", "prey", "mean_energy", "max_generation", "change_from_previous_window",
                            "weight_spread"] + list(self.pred_stats) + P_ACTIONS)
            w.writerow([self.tick, self.n_pred, self.population,
                        round(float(P["e"].mean()), 3) if self.n_pred else 0,
                        int(P["gen"].max()) if self.n_pred else 0, change, round(spread, 4)]
                       + list(self.pred_stats.values()) + [round(float(x), 4) for x in share])
        self._prev_pred_share = share
        self._pred_act_counts[:] = 0

    # ---------------------------------------------------------------- save / load
    def _checkpoint_extra(self):
        return dict(pred=self.pred, pred_pool=self.pred_pool, pred_stats=self.pred_stats, pred_bank=self.pred_bank)

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.pred = d.get("pred") or dict(x=np.zeros(0, np.int64), y=np.zeros(0, np.int64), e=np.zeros(0),
                                         age=np.zeros(0, np.int64), gen=np.zeros(0, np.int64))
        sim.pred_pool = d.get("pred_pool")
        sim.pred_stats = d.get("pred_stats", dict(kills=0, attacks=0, births=0, deaths=0, immigrants=0))
        sim.pred_bank = d.get("pred_bank")
        sim._pred_act_counts = np.zeros(P_OUT, np.int64)
        sim._prev_pred_share = None
        sim.pred_sense = None
        return sim
