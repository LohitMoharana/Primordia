"""
Evolvable actions: creatures can evolve new actions of their own, built from the basic ones.

Why: Soros & Stanley (2014) condition 4, "the potential complexity of the phenotype should not
be limited by its representation", and Taylor (2015), "individuals capable of producing more
complex offspring". Open brains (brains.py) showed that brain size is not the limit in Primordia:
brains do not grow when allowed to, because every creature can still only choose among the same
14 actions. Here the repertoire itself can grow.

A creature owns up to max_macros sequence-actions (a "macro"), each a chain of 2 to max_len basic
actions (any basic action except reproduce). Each owned macro is one more output of the brain.
When the brain picks it, the creature carries out the whole chain over the next ticks (its brain
is not consulted until the chain ends); a chain is the creature's own behaviour program, beyond
anything the 14 fixed actions can express in one choice.

Inheritance (own random stream rng_act; a child copies its parent's macros, then):
  new        p_new   a new macro of 2 random steps (output bias -2: rare at first, like the new
                     actions added in earlier stages)
  duplicate  p_dup   a copy of an existing macro, with its brain wiring (gene duplication: the
                     copy is free to change while the original keeps working)
  edit       p_edit  one macro changes one step, gains a step at the end, or loses one
  delete     p_del   the newest macro is removed
Cost: each owned macro costs macro_cost energy per tick (so macros spread only if they pay).

Control (actions_wired=False): macros are created, copied, edited and paid for exactly the same
way, but the brain can never choose them (their outputs are held at bias -30).

The brain has room for max_macros extra outputs from the start; unowned ones have zero weights and
bias -30, so they are never chosen (exp(-30) ~ 1e-13 against the 14 basic actions).
"""
import numpy as np

from engine import ACTIONS, A, MAX_AGE
from predators import PredSim
from ways import WaysSim

N_BASE = len(ACTIONS)                      # 14
STEP_CHOICES = np.array([i for i in range(N_BASE) if i != A["reproduce"]])
OFF_BIAS = -30.0


def _pad_out(f, M):
    f = dict(f)
    n, H, O = np.asarray(f["w2"]).shape
    f["w2"] = np.concatenate([np.asarray(f["w2"]), np.zeros((n, H, M), np.asarray(f["w2"]).dtype)], 2)
    f["b2"] = np.concatenate([np.asarray(f["b2"]), np.full((n, M), OFF_BIAS, np.asarray(f["b2"]).dtype)], 1)
    return f


class ActionMixin:
    _FIELDS_EXTRA = ["mac", "mac_len", "n_mac", "mac_cur", "mac_pos"]
    _METER_EXTRA = ["_mac_stats"]

    def __init__(self, *args, actions_evolve=True, actions_wired=True, max_macros=16, max_len=8,
                 p_new=0.02, p_dup=0.01, p_edit=0.05, p_del=0.01, macro_cost=0.0002, **kw):
        M = int(max_macros)
        self._M, self._L = M, int(max_len)
        if kw.get("founders") is not None:
            kw["founders"] = _pad_out(kw["founders"], M)
        kw["n_actions"] = N_BASE + M
        super().__init__(*args, **kw)
        seed = int(kw.get("seed", args[0] if args else 1))
        self.rng_act = np.random.default_rng([seed, 5151])
        self.cfg.update(actions_evolve=bool(actions_evolve), actions_wired=bool(actions_wired), max_macros=M,
                        max_len=int(max_len), p_new=float(p_new), p_dup=float(p_dup), p_edit=float(p_edit),
                        p_del=float(p_del), macro_cost=float(macro_cost))
        self._mac_stats = dict(started=0, steps=0, acts=0)
        idx = np.flatnonzero(self.alive)
        self.n_mac[idx] = 0
        self.mac_cur[idx] = -1
        self._fix_outputs(idx)

    def _alloc(self, cap):
        super()._alloc(cap)
        cfg = getattr(self, "cfg", {}) or {}
        M = getattr(self, "_M", None) or cfg.get("max_macros", 16)
        L = getattr(self, "_L", None) or cfg.get("max_len", 8)
        self._M, self._L = int(M), int(L)
        self.mac = np.full((cap, M, L), -1, np.int64)
        self.mac_len = np.zeros((cap, M), np.int64)
        self.n_mac = np.zeros(cap, np.int64)
        self.mac_cur = np.full(cap, -1, np.int64)
        self.mac_pos = np.zeros(cap, np.int64)

    def _fix_outputs(self, s):
        """Unowned macro outputs (and, in the control, all macro outputs): zero weights, bias -30."""
        if not len(s):
            return
        M = self._M
        off = np.arange(M)[None, :] >= self.n_mac[s][:, None]
        if not self.cfg.get("actions_wired", True):
            off[:] = True
        cols = N_BASE + np.arange(M)
        w2 = self.w2[s][:, :, cols]
        w2[np.broadcast_to(off[:, None, :], w2.shape)] = 0.0
        self.w2[np.ix_(s, np.arange(self.H), cols)] = w2
        b2 = self.b2[s][:, cols]
        b2[off] = OFF_BIAS
        self.b2[np.ix_(s, cols)] = b2

    # ------------------------------------------------------------------ choosing and running macros
    def _choose_actions(self, idx, act):
        act = super()._choose_actions(idx, act)
        cur = self.mac_cur[idx]
        busy = cur >= 0
        st = self._mac_stats
        st["acts"] += len(idx)
        # creatures in the middle of a chain: next step, brain not consulted
        if busy.any():
            b = idx[busy]
            pos = self.mac_pos[b]
            act[busy] = self.mac[b, cur[busy], pos]
            pos = pos + 1
            done = pos >= self.mac_len[b, cur[busy]]
            self.mac_pos[b] = pos
            self.mac_cur[b[done]] = -1
            st["steps"] += int(busy.sum())
        # creatures that just chose a macro: start it
        start = (~busy) & (act >= N_BASE)
        if start.any():
            s = idx[start]
            k = act[start] - N_BASE
            act[start] = self.mac[s, k, 0]
            more = self.mac_len[s, k] > 1
            self.mac_cur[s[more]] = k[more]
            self.mac_pos[s[more]] = 1
            st["started"] += int(start.sum())
            st["steps"] += int(start.sum())
        return act

    def _after_act(self, idx, act, x0, y0, e, keys, dead):
        super()._after_act(idx, act, x0, y0, e, keys, dead)
        e -= self.cfg["macro_cost"] * self.n_mac[idx]
        newly = (dead == 0) & ((e <= 0) | (self.age[idx] >= MAX_AGE))
        dead[newly] = 1

    # ------------------------------------------------------------------ inheritance
    def _spawn_children(self, par):
        s = super()._spawn_children(par)
        c, r, M, L = self.cfg, self.rng_act, self._M, self._L
        self.mac[s] = self.mac[par]
        self.mac_len[s] = self.mac_len[par]
        self.n_mac[s] = self.n_mac[par]
        self.mac_cur[s] = -1
        self.mac_pos[s] = 0
        if c["actions_evolve"]:
            n = len(s)
            u = r.random((n, 4))
            for i in np.flatnonzero((u < [c["p_new"], c["p_dup"], c["p_edit"], c["p_del"]]).any(1)):
                a = s[i]
                if u[i, 3] < c["p_del"] and self.n_mac[a] > 0:            # delete the newest
                    k = self.n_mac[a] - 1
                    self.mac[a, k] = -1; self.mac_len[a, k] = 0; self.n_mac[a] = k
                if u[i, 2] < c["p_edit"] and self.n_mac[a] > 0:           # edit one
                    k = r.integers(self.n_mac[a]); ln = self.mac_len[a, k]
                    op = r.integers(3)
                    if op == 0:
                        self.mac[a, k, r.integers(ln)] = r.choice(STEP_CHOICES)
                    elif op == 1 and ln < L:
                        self.mac[a, k, ln] = r.choice(STEP_CHOICES); self.mac_len[a, k] = ln + 1
                    elif op == 2 and ln > 2:
                        self.mac[a, k, ln - 1] = -1; self.mac_len[a, k] = ln - 1
                if u[i, 1] < c["p_dup"] and 0 < self.n_mac[a] < M:         # duplicate one, wiring too
                    k, j = r.integers(self.n_mac[a]), self.n_mac[a]
                    self.mac[a, j] = self.mac[a, k]; self.mac_len[a, j] = self.mac_len[a, k]
                    self.w2[a, :, N_BASE + j] = self.w2[a, :, N_BASE + k]
                    self.b2[a, N_BASE + j] = self.b2[a, N_BASE + k]
                    self.n_mac[a] = j + 1
                if u[i, 0] < c["p_new"] and self.n_mac[a] < M:             # a brand-new one
                    j = self.n_mac[a]
                    self.mac[a, j, :2] = r.choice(STEP_CHOICES, 2); self.mac[a, j, 2:] = -1
                    self.mac_len[a, j] = 2
                    self.w2[a, :, N_BASE + j] = r.normal(0, 0.05, self.H)
                    self.b2[a, N_BASE + j] = -2.0
                    self.n_mac[a] = j + 1
        self._fix_outputs(s)
        return s

    def _checkpoint_extra(self):
        d = super()._checkpoint_extra()
        d.update(rng_act_state=self.rng_act.bit_generator.state)
        return d

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        import pickle
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.rng_act = np.random.default_rng()
        sim.rng_act.bit_generator.state = d["rng_act_state"]
        if getattr(sim, "_mac_stats", None) is None:
            sim._mac_stats = dict(started=0, steps=0, acts=0)
        return sim

    # ------------------------------------------------------------------ statistics
    def macro_summary(self):
        """Macros in the living population: owned, length, and how much behaviour runs through them."""
        idx = np.flatnonzero(self.alive)
        st = self._mac_stats
        own = self.n_mac[idx]
        lens = self.mac_len[idx][self.mac_len[idx] > 0]
        out = dict(macros_owned=float(own.mean()) if len(idx) else 0.0,
                   macro_len_mean=float(lens.mean()) if len(lens) else 0.0,
                   macro_len_max=int(lens.max()) if len(lens) else 0,
                   macro_share=st["steps"] / max(1, st["acts"]))
        self._mac_stats = dict(started=0, steps=0, acts=0)
        return out


class WaysActionSim(ActionMixin, WaysSim):
    _FIELDS = WaysSim._FIELDS + ActionMixin._FIELDS_EXTRA
    _METER = WaysSim._METER + ActionMixin._METER_EXTRA


class PredActionSim(ActionMixin, PredSim):
    _FIELDS = PredSim._FIELDS + ActionMixin._FIELDS_EXTRA
    _METER = PredSim._METER + ActionMixin._METER_EXTRA
