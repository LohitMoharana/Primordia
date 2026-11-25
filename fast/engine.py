"""
Primordia fast engine: the same world rules as stages/13_cofood, vectorised.

All agents live in flat NumPy arrays (positions, energy, age, ...) and all
brains are stacked weight tensors, so a tick is a handful of array operations
instead of a Python loop over agents. Rules copied from the stage-13 code
(world.py, agent.py, sim.py): same costs, gains, food dynamics, seasons,
hazards, signals, sharing, reproduction, mutation and shared food.

What changes: in the old engine agents acted one at a time in a random order,
so a later agent could see food an earlier one had just eaten. Here every
agent senses the world as it is at the start of the tick, then all act at
once. Where order matters, it is reproduced explicitly:
  * several agents eating in one cell take food in a random order, each up to
    EAT_GAIN, exactly as the sequential min() did;
  * shared food in a cell is split among its eaters if there are 2+;
  * a share gift goes to a random other agent within radius 1.
So results match the old engine statistically, not bit for bit.

Features supported (the kept base world + shared food): social sensing,
communication, seasons, hazards, shared food. Inputs keep the stage-13 layout
(227 values; the kin, predator, marker and second-symbol patches are present
and zero), so stage-13 founders and brains load unchanged.
"""
import csv
import os
import pickle
import time

import numpy as np

try:
    from kernels import HAVE_NUMBA, sense_think, act_phase, prepare, resolve_cofood, food_decay, food_regrow, food_regrow_map
except ImportError:  # running from another directory without kernels on the path
    from .kernels import HAVE_NUMBA, sense_think, act_phase, prepare, resolve_cofood, food_decay, food_regrow, food_regrow_map  # type: ignore

# ---- constants copied from stages/13_cofood (agent.py, world.py, brain.py) ----
MAX_ENERGY = 10.0
START_ENERGY = 5.0
MAX_AGE = 500
MOVE_COST = 0.03
IDLE_COST = 0.01
SIGNAL_COST = 0.015
MARK_COST = 0.015
HAZARD_DRAIN = 0.15
SHARE_AMOUNT = 0.5
SHARE_COST = 0.02
SHARE_MULTIPLIER = 2.0  # receiver gets 2x what the giver pays (kin-sharing stage). This creates energy:
# two creatures passing gifts back and forth gain energy from nothing (found by evolution in the
# building world, 25 Nov). New experiments set cfg share_multiplier = 1.0 (gifts break even);
# older modes keep 2.0 so they still reproduce.
EAT_GAIN = 0.5
REPRODUCE_COST = 4.0
REPRODUCE_MIN_ENERGY = 6.0
SENSE_RADIUS = 2
N_SYMBOLS = 4
INPUT_SIZE = 25 * 9 + 2
HIDDEN_SIZE = 24
ACTIONS = ["stay", "N", "S", "E", "W", "eat", "reproduce"] + \
          [f"signal_{i}" for i in range(1, N_SYMBOLS + 1)] + ["share", "mate", "mark"]
OUTPUT_SIZE = len(ACTIONS)
A = {a: i for i, a in enumerate(ACTIONS)}
SIG0 = A["signal_1"]  # signal_k has index SIG0 + k - 1
MUT_RATE, MUT_STRENGTH = 0.1, 0.15

LOG_EVERY = 10
CHECKPOINT_EVERY = 2500

# input slot offsets (each patch is 25 values)
P_FOOD, P_OCC, P_SIG, P_HAZ, P_KIN, P_PRED, P_MARK, P_SIG2, P_COFOOD = [25 * i for i in range(9)]
P_EXTRA = 225


class FastSim:
    def __init__(self, seed=1, size=48, out_dir=None, founders=None, init_population=80,
                 social_sensing=True, communication=True, seasons=True, hazards=True,
                 cofood=False, food_regrow_prob=0.02, food_decay_prob=0.01, max_food=1.0,
                 season_period=600, season_amplitude=0.9, hazard_fraction=0.03,
                 cofood_spawn_prob=0.5, cofood_value=4.0, cofood_life=150,
                 dtype=np.float32, checkpoint_every=CHECKPOINT_EVERY,
                 hidden=HIDDEN_SIZE, n_actions=OUTPUT_SIZE, init_inputs=INPUT_SIZE,
                 memory=False, language=False, food_mode="uniform", n_patches=16, patch_sigma=4.0,
                 patch_speed=0.03, patch_turn=0.05, patch_peak=0.057, founders_near_patches=False,
                 patch_life=0, zero_memory_weights=False, walls=False, wall_spacing=32, door_width=6,
                 shelter_frac=0.0, share_multiplier=None):
        # hidden / n_actions / init_inputs: brain shape of the stage being modelled
        # (earlier stages had fewer actions, fewer inputs, stage 1 had 16 hidden units).
        # Actions are always a prefix of ACTIONS, so a stage with 7 actions can
        # only ever choose stay/N/S/E/W/eat/reproduce.
        self.H, self.O = int(hidden), int(n_actions)
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.dtype = dtype
        self.cfg = dict(social_sensing=social_sensing, communication=communication,
                        seasons=seasons, hazards=hazards, cofood=cofood,
                        food_regrow_prob=food_regrow_prob, food_decay_prob=food_decay_prob,
                        max_food=max_food, season_period=season_period,
                        season_amplitude=season_amplitude, hazard_fraction=hazard_fraction,
                        cofood_spawn_prob=cofood_spawn_prob, cofood_value=cofood_value,
                        cofood_life=cofood_life, hidden=int(hidden), n_actions=int(n_actions),
                        init_inputs=int(init_inputs), memory=bool(memory), language=bool(language), food_mode=food_mode,
                        n_patches=int(n_patches), patch_sigma=float(patch_sigma),
                        patch_speed=float(patch_speed), patch_turn=float(patch_turn),
                        patch_peak=float(patch_peak), patch_life=float(patch_life), walls=bool(walls),
                        wall_spacing=int(wall_spacing), door_width=int(door_width))
        if shelter_frac:
            self.cfg["shelter_frac"] = float(shelter_frac)
        if share_multiplier is not None:  # absent = old rule (2.0), so older modes reproduce
            self.cfg["share_multiplier"] = float(share_multiplier)
        self.checkpoint_every = checkpoint_every
        self.use_numba = HAVE_NUMBA
        self._complexity = (None, 0.0)
        r = self.rng
        # world, same initialisation as World.__init__
        # TERRAIN: walls along every wall_spacing-th row and column split the world
        # into rooms; each wall segment between two crossings has one doorway
        # door_width cells wide at a random spot. Walls block movement, grow no
        # food, and agents sense them (inputs 125-149).
        self.wall = np.zeros((size, size))
        if walls:
            for k in range(0, size, wall_spacing):
                self.wall[k, :] = 1.0
                self.wall[:, k] = 1.0
            for k in range(0, size, wall_spacing):
                for s0 in range(0, size, wall_spacing):
                    d = int(r.integers(s0 + 1, s0 + wall_spacing - door_width))
                    self.wall[k, d:d + door_width] = 0.0   # doorway in the horizontal wall
                    d = int(r.integers(s0 + 1, s0 + wall_spacing - door_width))
                    self.wall[d:d + door_width, k] = 0.0   # doorway in the vertical wall
        # SHELTERS (arms race round 2): scattered cells where predators cannot catch prey;
        # no food grows there. Drawn from their own random stream (so the rest of the world
        # is the same as without shelters) and sensed in inputs 100-124 (the unused kin slot).
        self.shelter = np.zeros((size, size))
        if shelter_frac:
            self.shelter = (np.random.default_rng([int(seed), 7919]).random((size, size)) < shelter_frac) * 1.0
            self.shelter[self.wall > 0] = 0.0
        self._has_shelter = bool(self.shelter.any())
        self.food = r.uniform(0, max_food, (size, size))
        self.food *= (r.random((size, size)) < 0.3)
        if food_mode == "patches":
            # SPARSE WORLD: food regrows only inside n_patches drifting clusters.
            # Each cluster is a Gaussian bump of regrowth probability (peak
            # patch_peak at its centre, width patch_sigma cells); centres move at
            # patch_speed cells/tick along a slowly turning heading. Between
            # clusters the ground is bare, so most of the world is out of sight
            # of food (sensing radius stays 2).
            self.patch_xy = r.uniform(0, size, (n_patches, 2))
            self.patch_dir = r.uniform(0, 2 * np.pi, n_patches)
            self._update_patch_map()
            self.food = max_food * (r.random((size, size)) < self.regrow_map / patch_peak * 0.5)
        self.food[self.wall > 0] = 0.0
        self.food[self.shelter > 0] = 0.0
        self.hazard = np.zeros((size, size))
        self.cofood_grid = np.zeros((size, size))
        self.cofood_age = np.zeros((size, size))
        self.cofood_stats = {"spawned": 0, "eaten": 0, "expired": 0, "solo_attempts": 0, "co_eaters": 0}
        self.t = 0       # world clock (World.t)
        self.tick = 0    # simulation tick
        self.next_id = 0
        self._alloc(256)
        if founders is not None and zero_memory_weights:
            # fair memory test: founders start with NO memory wiring (wh = 0); only
            # mutation and selection can build it up, instead of inheriting noise
            founders = dict(founders); founders["wh"] = np.zeros_like(founders["wh"])
        if founders is not None:
            if founders_near_patches and food_mode == "patches":
                founders = dict(founders)
                k = r.integers(0, n_patches, len(founders["x"]))
                founders["x"] = np.round(self.patch_xy[k, 0] + r.normal(0, patch_sigma, len(k))).astype(int) % size
                founders["y"] = np.round(self.patch_xy[k, 1] + r.normal(0, patch_sigma, len(k))).astype(int) % size
            if self.wall.any():  # nobody starts inside a wall
                founders = dict(founders)
                fx, fy = np.asarray(founders["x"]) % size, np.asarray(founders["y"]) % size
                for _ in range(4):  # step diagonally off the wall (crossings may need two steps)
                    on = self.wall[fx, fy] > 0
                    fx, fy = np.where(on, (fx + 1) % size, fx), np.where(on, (fy + 1) % size, fy)
                founders["x"], founders["y"] = fx, fy
            self.add_agents(**founders)
        else:
            n = init_population
            H, O = self.H, self.O
            self.add_agents(
                x=r.integers(0, size, n), y=r.integers(0, size, n),
                w1=r.normal(0, np.sqrt(2.0 / init_inputs), (n, INPUT_SIZE, H)),
                b1=np.zeros((n, H)),
                w2=r.normal(0, np.sqrt(2.0 / H), (n, H, O)),
                b2=np.zeros((n, O)),
                wh=r.normal(0, 0.1 / np.sqrt(H), (n, H, H)))
        # patch offsets: local_view flattens grid[x+dx, y+dy] with dx outer, dy inner
        d = np.arange(-SENSE_RADIUS, SENSE_RADIUS + 1)
        self._ox = np.repeat(d, 5)
        self._oy = np.tile(d, 5)
        self.out_dir = out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self.log_path = os.path.join(out_dir, "log.csv")
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", newline="") as f:
                    csv.writer(f).writerow(["tick", "population", "mean_energy", "mean_age",
                                            "max_generation", "mean_complexity", "total_food"])

    # ------------------------------------------------------------------ pool
    def _alloc(self, cap):
        dt = self.dtype
        self.cap = cap
        self.alive = np.zeros(cap, bool)
        self.x = np.zeros(cap, np.int64)
        self.y = np.zeros(cap, np.int64)
        self.energy = np.zeros(cap)
        self.age = np.zeros(cap, np.int64)
        self.gen = np.zeros(cap, np.int64)
        self.aid = np.full(cap, -1, np.int64)
        self.parent = np.full(cap, -1, np.int64)
        self.last_action = np.full(cap, -1, np.int64)
        H, O = self.H, self.O
        self.w1 = np.zeros((cap, INPUT_SIZE, H), dt)
        self.b1 = np.zeros((cap, H), dt)
        self.w2 = np.zeros((cap, H, O), dt)
        self.b2 = np.zeros((cap, O), dt)
        self.wh = np.zeros((cap, H, H), dt)
        self.hstate = np.zeros((cap, H), dt)  # working memory (used only with memory=True)
        self.ws = np.zeros((cap, H, N_SYMBOLS), dt)  # second-symbol head (language)
        self.bs = np.zeros((cap, N_SYMBOLS), dt)
        self.last_sym2 = np.zeros(cap, np.int64)
        self.lineage = np.full(cap, -1, np.int64)  # id of the founding ancestor

    # counters behind novelty.csv / log.csv (and subclasses' logs), saved with checkpoints
    _METER = ["_act_counts", "_prev_share", "_complexity"]

    _FIELDS = ["alive", "x", "y", "energy", "age", "gen", "aid", "parent", "last_action",
               "w1", "b1", "w2", "b2", "wh", "hstate", "ws", "bs", "last_sym2", "lineage"]

    def _grow(self, need):
        new_cap = max(need, int(self.cap * 1.5) + 64)
        old = {k: getattr(self, k) for k in self._FIELDS}
        self._alloc(new_cap)
        for k, v in old.items():
            getattr(self, k)[:len(v)] = v

    def _compact(self):
        idx = np.flatnonzero(self.alive)
        for k in self._FIELDS:
            arr = getattr(self, k)
            arr[:len(idx)] = arr[idx]
        self.alive[:] = False
        self.alive[:len(idx)] = True

    def add_agents(self, x, y, w1, b1, w2, b2, wh, energy=None, gen=None, parent=None, ws=None, bs=None,
                   lineage=None):
        n = len(x)
        free = np.flatnonzero(~self.alive)
        if len(free) < n:
            self._grow(self.cap + n - len(free))
            free = np.flatnonzero(~self.alive)
        s = free[:n]
        self.alive[s] = True
        self.x[s], self.y[s] = np.asarray(x) % self.size, np.asarray(y) % self.size
        self.energy[s] = START_ENERGY if energy is None else energy
        self.age[s] = 0
        self.gen[s] = 0 if gen is None else gen
        self.parent[s] = -1 if parent is None else parent
        self.aid[s] = np.arange(self.next_id, self.next_id + n)
        self.lineage[s] = self.aid[s] if lineage is None else lineage  # founders start their own lineage
        if getattr(self, "_births", None) is not None and parent is not None:
            self._births.append(np.stack([np.full(n, self.tick + 1), self.aid[s], np.asarray(parent),
                                          np.asarray(gen), self.lineage[s]], 1))
        self.next_id += n
        self.last_action[s] = -1
        self.w1[s], self.b1[s], self.w2[s], self.b2[s], self.wh[s] = w1, b1, w2, b2, wh
        self.hstate[s] = 0.0  # memory starts empty at birth
        # second-symbol head: new brains start near-uniform (tiny weights), as in stage 12
        self.ws[s] = ws if ws is not None else self.rng.normal(0, 0.05, (n, self.H, N_SYMBOLS))
        self.bs[s] = bs if bs is not None else 0.0
        self.last_sym2[s] = 0
        return s

    @property
    def population(self):
        return int(self.alive.sum())

    # ------------------------------------------------------------------ world
    def _update_patch_map(self):
        c, S = self.cfg, self.size
        g = np.arange(S)
        sig2 = 2 * c["patch_sigma"] ** 2
        def bumps(centres):  # (K, S): Gaussian along one axis, toroidal distance
            d = np.abs(g[None, :] - centres[:, None])
            d = np.minimum(d, S - d)
            return np.exp(-d ** 2 / sig2)
        gx, gy = bumps(self.patch_xy[:, 0]), bumps(self.patch_xy[:, 1])
        self.regrow_map = np.minimum(c["patch_peak"] * np.einsum("kx,ky->xy", gx, gy), 1.0)
        if getattr(self, "wall", None) is not None and self.wall.any():
            self.regrow_map = self.regrow_map * (1.0 - self.wall)  # no food grows on walls
        if getattr(self, "shelter", None) is not None and self.shelter.any():
            self.regrow_map = self.regrow_map * (1.0 - self.shelter)  # nor in shelters

    def _sample_patch_cell(self):
        cum = np.cumsum(self.regrow_map.ravel())
        k = int(np.searchsorted(cum, self.rng.random() * cum[-1]))
        return k // self.size, k % self.size

    def _move_patches(self):
        c, r = self.cfg, self.rng
        self.patch_dir += r.normal(0, c["patch_turn"], len(self.patch_dir))
        self.patch_xy[:, 0] = (self.patch_xy[:, 0] + c["patch_speed"] * np.cos(self.patch_dir)) % self.size
        self.patch_xy[:, 1] = (self.patch_xy[:, 1] + c["patch_speed"] * np.sin(self.patch_dir)) % self.size
        moved = False
        if c.get("patch_life", 0) > 0:
            # each cluster vanishes with probability 1/patch_life per tick and reappears
            # at a random spot: where food is cannot be learned once and for all
            gone = r.random(len(self.patch_dir)) < 1.0 / c["patch_life"]
            if gone.any():
                self.patch_xy[gone] = r.uniform(0, self.size, (int(gone.sum()), 2))
                moved = True
        if self.t % 10 == 0 or moved:
            self._update_patch_map()

    def _harsh(self):
        return 1.0  # ramp_ticks = 0 in stages 6-13

    def _world_step(self, foods=None):
        """One world tick. foods: optional (k, S, S) stack of private food grids
        (used by the survival test); every grid gets the same random draws."""
        c, r, S = self.cfg, self.rng, self.size
        F = self.food[None] if foods is None else foods
        fast = foods is None and self.use_numba
        dr = r.random((S, S))
        if fast:
            food_decay(self.food, dr, c["food_decay_prob"])
        else:
            F[:, dr < c["food_decay_prob"]] *= 0.5
        self.t += 1
        if c["hazards"] and (self.t == 1 or self.t % c["season_period"] == 0):
            self.hazard = (r.random((S, S)) < c["hazard_fraction"] * self._harsh()).astype(float)
        self._cofood_step(F)
        p = c["food_regrow_prob"]
        if c["seasons"]:
            p = p * (1 + c["season_amplitude"] * np.sin(2 * np.pi * self.t / c["season_period"]))
        rr = r.random((S, S))
        if c.get("food_mode", "uniform") == "patches":
            self._move_patches()
            factor = p / c["food_regrow_prob"]  # seasons scale the whole map
            if fast:
                food_regrow_map(self.food, rr, self.regrow_map, factor, c["max_food"])
            else:
                F[(F < 0.05) & (rr < self.regrow_map * factor)[None]] = c["max_food"]
                np.clip(F, 0, c["max_food"], out=F)
        elif fast:
            food_regrow(self.food, rr, p, c["max_food"])
        else:
            F[(F < 0.05) & (rr < p)[None]] = c["max_food"]
            np.clip(F, 0, c["max_food"], out=F)
        if self._has_shelter:
            F[:, self.shelter > 0] = 0.0  # nothing grows in shelters (uniform mode; patches mode masks the map)

    def _cofood_step(self, F):
        c, r, S = self.cfg, self.rng, self.size
        if c["cofood_spawn_prob"] <= 0:
            return
        if c["cofood"]:
            live = self.cofood_grid > 0
            self.cofood_age[live] += 1
            old = live & (self.cofood_age > c["cofood_life"])
            self.cofood_stats["expired"] += int(old.sum())
            self.cofood_grid[old] = 0
        if r.random() < c["cofood_spawn_prob"]:
            self.cofood_stats["spawned"] += 1
            if c.get("food_mode", "uniform") == "patches":
                # sparse world: shared food appears where food grows (inside the clusters)
                x, y = self._sample_patch_cell()
            else:
                x, y = int(r.integers(S)), int(r.integers(S))
            if c["cofood"]:
                self.cofood_grid[x, y] += c["cofood_value"]
                self.cofood_age[x, y] = 0
            else:
                # add exactly cofood_value of ordinary food, topping up random cells
                # (draws until the grid that needs the most cells is satisfied;
                # with one grid this is exactly the old loop)
                left = np.full(len(F), c["cofood_value"])
                while (left > 1e-9).any():
                    if c.get("food_mode", "uniform") == "patches":
                        cx, cy = self._sample_patch_cell()
                    else:
                        cx, cy = int(r.integers(S)), int(r.integers(S))
                    add = np.minimum(c["max_food"] - F[:, cx, cy], left)
                    F[:, cx, cy] += add
                    left -= add

    # ------------------------------------------------------------------ agents
    def _patches(self, grid, xs, ys):
        S = self.size
        return grid[(xs[:, None] + self._ox) % S, (ys[:, None] + self._oy) % S]

    def sense(self, idx, occ, sig, sig2=None):
        """Inputs for agents idx, in the stage-13 layout (n x 227)."""
        S = self.size
        xs, ys = self.x[idx], self.y[idx]
        flat = ((xs[:, None] + self._ox) % S) * S + (ys[:, None] + self._oy) % S  # n x 25
        cof_seen = self.cofood_grid if self.cfg.get("cofood_visible", True) else np.zeros_like(self.cofood_grid)
        grids = np.stack([self.food, occ, sig, self.hazard, cof_seen,
                          np.zeros_like(occ) if sig2 is None else sig2]).reshape(6, -1)
        g = grids[:, flat]  # 5 x n x 25, one gather for all grids
        n = len(idx)
        X = np.zeros((n, INPUT_SIZE), self.dtype)
        X[:, P_FOOD:P_FOOD + 25] = g[0]
        if self.cfg["social_sensing"]:
            o = g[1]
            o[:, 12] = np.maximum(0.0, o[:, 12] - 1)  # don't count yourself
            X[:, P_OCC:P_OCC + 25] = o
        X[:, P_SIG:P_SIG + 25] = g[2]
        X[:, P_HAZ:P_HAZ + 25] = g[3]
        X[:, P_COFOOD:P_COFOOD + 25] = g[4] > 0
        X[:, P_SIG2:P_SIG2 + 25] = g[5]
        sg = self._pred_slot_grid()
        if sg.any():
            X[:, P_PRED:P_PRED + 25] = sg.ravel()[flat] > 0
        if self._has_shelter:
            X[:, P_KIN:P_KIN + 25] = self.shelter.ravel()[flat] > 0
        mg, has_mark = self._mark_slot_args()
        if has_mark:
            X[:, P_MARK:P_MARK + 25] = mg.ravel()[flat]
        X[:, P_EXTRA] = self.energy[idx] / MAX_ENERGY
        X[:, P_EXTRA + 1] = self.age[idx] / MAX_AGE
        return X

    def think(self, idx, X):
        """Batched forward pass (memory off: no carried state) and action sampling.

        Works directly on the stored weight arrays over slots [0, hi) instead of
        copying each alive agent's brain every tick (dead slots in that range are
        computed and ignored; the pool is compacted when they pile up). Input
        rows that are always zero in these stages (kin, predator, marker, second
        symbol: slots 100-199) are skipped.
        """
        hi = int(idx[-1]) + 1
        Xf = np.zeros((hi, INPUT_SIZE), self.dtype)
        Xf[idx] = X
        w1 = self.w1[:hi]
        pre = np.einsum("ni,nih->nh", Xf[:, :P_KIN], w1[:, :P_KIN], optimize=False)
        pre += np.einsum("ni,nih->nh", Xf[:, P_SIG2:P_COFOOD], w1[:, P_SIG2:P_COFOOD], optimize=False)
        pre += np.einsum("ni,nih->nh", Xf[:, P_PRED:P_MARK], w1[:, P_PRED:P_MARK], optimize=False)
        pre += np.einsum("ni,nih->nh", Xf[:, P_COFOOD:], w1[:, P_COFOOD:], optimize=False)
        if self._has_shelter:
            pre += np.einsum("ni,nih->nh", Xf[:, P_KIN:P_PRED], w1[:, P_KIN:P_PRED], optimize=False)
        if self._mark_slot_args()[1]:
            pre += np.einsum("ni,nih->nh", Xf[:, P_MARK:P_SIG2], w1[:, P_MARK:P_SIG2], optimize=False)
        if self.cfg.get("memory", False):
            pre += np.einsum("ni,nij->nj", self.hstate[:hi], self.wh[:hi], optimize=False)
        h = np.tanh(pre + self.b1[:hi])
        if self.cfg.get("memory", False):
            self.hstate[idx] = h[idx]
        out = np.einsum("nh,nho->no", h, self.w2[:hi]) + self.b2[:hi]
        out = out[idx]
        out = out - out.max(axis=1, keepdims=True)
        p = np.exp(out.astype(np.float64))
        p /= p.sum(axis=1, keepdims=True)
        u = self.rng.random(len(idx))
        act = (np.cumsum(p, axis=1) < u[:, None]).sum(axis=1)
        return np.minimum(act, self.O - 1), h[idx]

    def step(self):
        r, S = self.rng, self.size
        self._world_step()
        idx = np.flatnonzero(self.alive)
        if len(idx) and len(idx) < 0.7 * (idx[-1] + 1) and idx[-1] > 256:
            self._compact()
            idx = np.flatnonzero(self.alive)
        n = len(idx)
        if n == 0:
            self.tick += 1
            return
        if self.use_numba:
            self._step_numba(idx)
            return
        # occupancy and last tick's signals, before anyone acts (same as the old engine)
        occ = np.zeros((S, S))
        np.add.at(occ, (self.x[idx], self.y[idx]), 1)
        sig = np.zeros((S, S))
        if self.cfg["communication"]:
            la = self.last_action[idx]
            s = (la >= SIG0) & (la < SIG0 + N_SYMBOLS)
            sig[self.x[idx][s], self.y[idx][s]] = la[s] - SIG0 + 1
        self.age[idx] += 1
        sig2 = np.zeros((S, S))
        if self.cfg["communication"] and self.cfg.get("language", False):
            la = self.last_action[idx]
            s = (la >= SIG0) & (la < SIG0 + N_SYMBOLS)
            sig2[self.x[idx][s], self.y[idx][s]] = self.last_sym2[idx][s]
        X = self.sense(idx, occ, sig, sig2)
        act, h = self.think(idx, X)
        # second symbol of each message (chosen whenever the agent signals; delivered only with language on)
        u2 = r.random(n)
        sym2 = np.zeros(n, np.int64)
        sg = np.flatnonzero((act >= SIG0) & (act < SIG0 + N_SYMBOLS))
        if len(sg):
            lg = np.einsum("nh,nhq->nq", h[sg].astype(np.float32), self.ws[idx[sg]]) + self.bs[idx[sg]]
            pq = np.exp((lg - lg.max(1, keepdims=True)).astype(np.float64)); pq /= pq.sum(1, keepdims=True)
            sym2[sg] = np.minimum((np.cumsum(pq, 1) < u2[sg, None]).sum(1), N_SYMBOLS - 1) + 1
        self.last_sym2[idx] = sym2
        self._count_actions(act)
        # random numbers for this tick's action phase, drawn the same way on both paths
        keys = r.random(n)   # eating order inside a shared cell
        su = r.random(n)     # which neighbour receives a gift
        e = self.energy[idx].copy()
        x0, y0 = self.x[idx].copy(), self.y[idx].copy()

        # idle-cost actions
        e[np.isin(act, [A["stay"], A["eat"], A["reproduce"], A["mate"]])] -= IDLE_COST
        # moves
        mv = (act >= A["N"]) & (act <= A["W"])
        dx = np.select([act == A["E"], act == A["W"]], [1, -1], 0)
        dy = np.select([act == A["S"], act == A["N"]], [1, -1], 0)
        nx, ny = (x0 + dx) % S, (y0 + dy) % S
        blocked = self.wall[nx, ny] > 0          # walls block movement (the move is still paid)
        self.x[idx] = np.where(blocked, x0, nx)
        self.y[idx] = np.where(blocked, y0, ny)
        e[mv] -= MOVE_COST
        # signals and marks cost energy (marks are not sensed in these stages)
        e[(act >= SIG0) & (act < SIG0 + N_SYMBOLS)] -= SIGNAL_COST
        e[act == A["mark"]] -= MARK_COST

        # eating: agents in one cell take food in a random order, up to EAT_GAIN each
        eat = np.flatnonzero(act == A["eat"])
        if len(eat):
            cell = x0[eat] * S + y0[eat]
            order = np.lexsort((keys[eat], cell))
            cell_s = cell[order]
            first = np.r_[True, cell_s[1:] != cell_s[:-1]]
            start = np.flatnonzero(first)
            rank = np.arange(len(eat)) - np.repeat(start, np.diff(np.r_[start, len(eat)]))
            avail = self.food.ravel()[cell_s]
            gain = np.clip(avail - EAT_GAIN * rank, 0.0, EAT_GAIN)
            g = np.empty(len(eat))
            g[order] = gain
            eff = self._eat_eff(idx)
            e[eat] += g if eff is None else g * eff[eat]
            np.add.at(self.food.ravel(), cell, -g)
            np.maximum(self.food, 0, out=self.food)

        # reproduction (energy after this tick's idle cost)
        rep = np.flatnonzero((act == A["reproduce"]) & (e >= REPRODUCE_MIN_ENERGY))
        e[rep] -= REPRODUCE_COST

        # sharing: pay the cost, then a gift if a neighbour exists and energy allows
        sh = np.flatnonzero(act == A["share"])
        gifts = None
        if len(sh):
            e[sh] -= SHARE_COST
            gifts = self._share_targets(idx, sh, x0, y0, e, su)
            if gifts is not None:
                e[gifts[0]] -= SHARE_AMOUNT  # giver pays now, before its own poison/death check

        # poison at the cell the agent ended up in; cap; deaths
        e[self.hazard[self.x[idx], self.y[idx]] > 0] -= HAZARD_DRAIN
        np.minimum(e, MAX_ENERGY, out=e)
        dead = (e <= 0) | (self.age[idx] >= MAX_AGE)
        dead_i = dead.astype(np.int64)
        self._after_act(idx, act, x0, y0, e, keys, dead_i)
        dead = dead_i == 1
        self.energy[idx] = e
        self.last_action[idx] = act
        self.alive[idx[dead]] = False

        # gifts and shared food go to agents still alive after their own action
        if gifts is not None:
            _, recv = gifts
            ok = self.alive[recv]
            np.add.at(self.energy, recv[ok], SHARE_AMOUNT * self.cfg.get("share_multiplier", SHARE_MULTIPLIER))
            np.minimum(self.energy, MAX_ENERGY, out=self.energy)
        if self.cfg["cofood"] and len(eat):
            self._resolve_cofood(idx[eat], x0[eat], y0[eat])

        # children, after everyone acted (they act from next tick)
        if len(rep):
            # the old engine adds the child even if the parent then died this tick
            self._spawn_children(idx[rep])
        self.tick += 1

    def _step_numba(self, idx):
        n, S, r = len(idx), self.size, self.rng
        occ = np.empty((S, S))
        sig = np.empty((S, S))
        xs = np.empty(n, np.int64)
        ys = np.empty(n, np.int64)
        ages = np.empty(n, np.int64)
        sig2 = np.empty((S, S))
        prepare(idx, self.x, self.y, self.last_action, self.age, S,
                bool(self.cfg["communication"]), SIG0, N_SYMBOLS, occ, sig, xs, ys, ages,
                bool(self.cfg.get("language", False)), self.last_sym2, sig2)
        act = np.empty(n, np.int64)
        sym2 = np.empty(n, np.int64)
        e = self.energy[idx]
        # cofood_visible=False: shared food still exists and can be eaten, but agents cannot see it
        cof_seen = self.cofood_grid if self.cfg.get("cofood_visible", True) else np.zeros_like(self.cofood_grid)
        sg = self._pred_slot_grid()
        sense_think(idx, xs, ys, e, ages.astype(np.float64), self.food, occ, sig, self.hazard,
                    cof_seen, bool(self.cfg["social_sensing"]), self.w1, self.b1,
                    self.w2, self.b2, r.random(n), MAX_ENERGY, float(MAX_AGE), act,
                    bool(self.cfg.get("memory", False)), self.wh, self.hstate,
                    sig2, self.ws, self.bs, r.random(n), SIG0, N_SYMBOLS, sym2,
                    sg, bool(sg.any()), self.shelter, bool(self._has_shelter),
                    *self._mark_slot_args(), *self._nact_args())
        self.last_sym2[idx] = sym2
        self._count_actions(act)
        keys = r.random(n)
        su = r.random(n)
        x0, y0 = xs.copy(), ys.copy()
        rep = np.empty(n, np.int64)
        dead = np.empty(n, np.int64)
        recv = np.empty(n, np.int64)
        act_phase(act, xs, ys, e, ages, self.food, self.hazard, keys, su, S,
                  A["stay"], A["N"], A["S"], A["E"], A["W"], A["eat"], A["reproduce"], A["mate"],
                  A["mark"], A["share"], SIG0, N_SYMBOLS, IDLE_COST, MOVE_COST, SIGNAL_COST,
                  MARK_COST, EAT_GAIN, REPRODUCE_MIN_ENERGY, REPRODUCE_COST, SHARE_COST,
                  SHARE_AMOUNT, HAZARD_DRAIN, MAX_ENERGY, MAX_AGE, rep, dead, recv, self.wall,
                  self._eat_eff(idx))
        self._after_act(idx, act, x0, y0, e, keys, dead)
        self.x[idx], self.y[idx] = xs, ys
        self.energy[idx] = e
        self.last_action[idx] = act
        self.alive[idx[dead == 1]] = False
        g = np.flatnonzero(recv >= 0)
        if len(g):
            rc = idx[recv[g]]
            ok = self.alive[rc]
            np.add.at(self.energy, rc[ok], SHARE_AMOUNT * self.cfg.get("share_multiplier", SHARE_MULTIPLIER))
            np.minimum(self.energy, MAX_ENERGY, out=self.energy)
        if self.cfg["cofood"]:
            eat = np.flatnonzero(act == A["eat"])
            if len(eat):
                st = np.zeros(3, np.int64)
                resolve_cofood(idx[eat], x0[eat], y0[eat], self.alive, self.energy,
                               self.cofood_grid, S, MAX_ENERGY, st)
                self.cofood_stats["eaten"] += int(st[0])
                self.cofood_stats["co_eaters"] += int(st[1])
                self.cofood_stats["solo_attempts"] += int(st[2])
        r_ = np.flatnonzero(rep == 1)
        if len(r_):
            self._spawn_children(idx[r_])
        self.tick += 1

    def _pred_slot_grid(self):
        """What agents sense in inputs 125-149: walls (terrain) and/or predators (predators.py)."""
        ps = getattr(self, "pred_sense", None)
        return self.wall if ps is None else np.maximum(self.wall, ps)

    def _checkpoint_extra(self):
        return {}

    # hooks for subclasses (ways.py); the defaults change nothing
    def _mark_slot_args(self):
        """Grid sensed in inputs 150-174 (the unused marker slot) and whether to read it."""
        if getattr(self, "_zero_grid", None) is None:
            self._zero_grid = np.zeros((self.size, self.size))
        return self._zero_grid, False

    _NO_NACT = np.zeros(1, np.int64)

    def _nact_args(self):
        """Per-agent number of hidden units in use (brains.py), and whether to apply it."""
        return self._NO_NACT, False

    def _eat_eff(self, idx):
        """Energy per unit of surface food eaten, per agent in idx (None / ones = 1)."""
        if self.use_numba:
            return np.ones(len(idx))
        return None

    def _after_act(self, idx, act, x0, y0, e, keys, dead):
        """Called after the action phase, before energies are written back (e, dead: local)."""
        return None

    def _count_actions(self, act):
        if getattr(self, "_act_counts", None) is None:
            self._act_counts = np.zeros(OUTPUT_SIZE, np.int64)
        self._act_counts += np.bincount(act, minlength=OUTPUT_SIZE)[:OUTPUT_SIZE]

    def _novelty(self):
        """Novelty meter, every 1000 ticks: how the population spends its actions,
        how much that changed since the last window (L1 distance, 0-2), how many
        founding lineages are still alive, and the brain-weight spread."""
        cnt = getattr(self, "_act_counts", None)
        if cnt is None or cnt.sum() == 0:
            return
        share = cnt / cnt.sum()
        prev = getattr(self, "_prev_share", None)
        change = "" if prev is None else round(float(np.abs(share - prev).sum()), 4)
        idx = np.flatnonzero(self.alive)
        lineages = len(np.unique(self.lineage[idx])) if len(idx) else 0
        smp = idx[:: max(1, len(idx) // 200)]
        spread = float(self.w2[smp].reshape(len(smp), -1).std(0).mean()) if len(smp) > 1 else 0.0
        path = os.path.join(self.out_dir, "novelty.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "change_from_previous_window", "lineages_alive", "weight_spread"] + ACTIONS)
            w.writerow([self.tick, change, lineages, round(spread, 4)] + [round(float(x), 4) for x in share])
        self._prev_share = share
        self._act_counts[:] = 0

    def _flush_births(self):
        if not getattr(self, "_births", None):
            return
        path = os.path.join(self.out_dir, "births.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "child_id", "parent_id", "generation", "lineage"])
            w.writerows(np.concatenate(self._births).tolist())
        self._births = []

    def _act_numba(self, idx, act, keys, su):
        n = len(idx)
        S = self.size
        xs, ys = self.x[idx].copy(), self.y[idx].copy()
        x0, y0 = xs.copy(), ys.copy()
        e = self.energy[idx].copy()
        rep = np.empty(n, np.int64)
        dead = np.empty(n, np.int64)
        recv = np.empty(n, np.int64)
        act_phase(act, xs, ys, e, self.age[idx], self.food, self.hazard, keys, su, S,
                  A["stay"], A["N"], A["S"], A["E"], A["W"], A["eat"], A["reproduce"], A["mate"],
                  A["mark"], A["share"], SIG0, N_SYMBOLS, IDLE_COST, MOVE_COST, SIGNAL_COST,
                  MARK_COST, EAT_GAIN, REPRODUCE_MIN_ENERGY, REPRODUCE_COST, SHARE_COST,
                  SHARE_AMOUNT, HAZARD_DRAIN, MAX_ENERGY, MAX_AGE, rep, dead, recv, self.wall,
                  self._eat_eff(idx))
        self.x[idx], self.y[idx] = xs, ys
        self.energy[idx] = e
        self.last_action[idx] = act
        self.alive[idx[dead == 1]] = False
        g = np.flatnonzero(recv >= 0)
        if len(g):
            rc = idx[recv[g]]
            ok = self.alive[rc]
            np.add.at(self.energy, rc[ok], SHARE_AMOUNT * self.cfg.get("share_multiplier", SHARE_MULTIPLIER))
            np.minimum(self.energy, MAX_ENERGY, out=self.energy)
        if self.cfg["cofood"]:
            eat = np.flatnonzero(act == A["eat"])
            if len(eat):
                self._resolve_cofood(idx[eat], x0[eat], y0[eat])
        r_ = np.flatnonzero(rep == 1)
        if len(r_):
            self._spawn_children(idx[r_])
        self.tick += 1

    def _share_targets(self, idx, sh, x0, y0, e, su):
        """Random other agent within radius 1 (positions at the start of the tick)."""
        S = self.size
        cell = x0 * S + y0
        order = np.argsort(cell, kind="stable")
        cs = cell[order]
        givers, recv = [], []
        for j in sh:
            if e[j] <= SHARE_AMOUNT:
                continue
            cands = []
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    c = ((x0[j] + ddx) % S) * S + (y0[j] + ddy) % S
                    lo, hi = np.searchsorted(cs, c), np.searchsorted(cs, c, side="right")
                    cands.extend(order[lo:hi].tolist())
            cands = [k for k in cands if k != j]
            if cands:
                givers.append(j)
                recv.append(idx[cands[int(su[j] * len(cands))]])
        if not givers:
            return None
        return np.array(givers), np.array(recv)  # (local giver indices, global receiver slots)

    def _resolve_cofood(self, eaters, ex, ey):
        S = self.size
        has = self.cofood_grid[ex, ey] > 0
        eaters, ex, ey = eaters[has], ex[has], ey[has]
        if not len(eaters):
            return
        ok = self.alive[eaters]
        cell = ex * S + ey
        cells, counts = np.unique(cell[ok], return_counts=True)
        all_cells = np.unique(cell)
        for c in all_cells:
            k = counts[np.searchsorted(cells, c)] if c in cells else 0
            if k >= 2:
                sel = eaters[ok & (cell == c)]
                share = self.cofood_grid.ravel()[c] / k
                self.energy[sel] = np.minimum(self.energy[sel] + share, MAX_ENERGY)
                self.cofood_grid.ravel()[c] = 0
                self.cofood_stats["eaten"] += 1
                self.cofood_stats["co_eaters"] += int(k)
            else:
                self.cofood_stats["solo_attempts"] += int(k)

    def _spawn_children(self, par):
        r = self.rng
        n = len(par)

        def mut(a):  # a is a fresh copy already (fancy indexing), so mutate it in place
            m = r.random(a.shape) < MUT_RATE
            a[m] += r.normal(0, MUT_STRENGTH, int(m.sum())).astype(a.dtype)
            return a

        return self.add_agents(self.x[par].copy(), self.y[par].copy(),
                        mut(self.w1[par]), mut(self.b1[par]), mut(self.w2[par]),
                        mut(self.b2[par]), mut(self.wh[par]), ws=mut(self.ws[par]), bs=mut(self.bs[par]),
                        energy=np.full(n, REPRODUCE_COST * 0.5),
                        gen=self.gen[par] + 1, parent=self.aid[par], lineage=self.lineage[par])

    # ------------------------------------------------------------------ io
    def log_row(self):
        idx = np.flatnonzero(self.alive)
        if not len(idx):
            return [self.tick, 0, 0, 0, 0, 0, float(self.food.sum())]
        # brain "complexity" (mean |weight|) is costly over every brain: refresh every 500 ticks
        if self._complexity[0] is None or self.tick - self._complexity[0] >= 500:
            smp = idx[:: max(1, len(idx) // 200)]  # every k-th agent: ~200 brains, cheap
            comp = np.abs(self.w1[smp]).mean(axis=(1, 2)) + np.abs(self.w2[smp]).mean(axis=(1, 2))
            self._complexity = (self.tick, float(comp.mean()))
        return [self.tick, len(idx), float(self.energy[idx].mean()), float(self.age[idx].mean()),
                int(self.gen[idx].max()), self._complexity[1], float(self.food.sum())]

    def checkpoint(self):
        if not self.out_dir:
            return
        idx = np.flatnonzero(self.alive)
        data = {k: getattr(self, k)[idx].copy() for k in self._FIELDS if k != "alive"}
        # where each agent sits in the pool, and the pool size: newborns fill free slots,
        # and slot order decides who draws which random number, so an exact resume needs it
        data.update(slot=idx.copy(), cap=self.cap)
        # running counters of the novelty meter and log, so a resumed run writes the same logs
        data.update(meter={k: getattr(self, k) for k in self._METER if getattr(self, k, None) is not None})
        data.update(tick=self.tick, t=self.t, next_id=self.next_id, cfg=self.cfg, size=self.size,
                    food=self.food, hazard=self.hazard, cofood_grid=self.cofood_grid,
                    cofood_age=self.cofood_age, cofood_stats=self.cofood_stats,
                    rng_state=self.rng.bit_generator.state, engine="fast-v1",
                    patch_xy=getattr(self, "patch_xy", None), patch_dir=getattr(self, "patch_dir", None),
                    regrow_map=getattr(self, "regrow_map", None), wall=self.wall, shelter=self.shelter)
        data.update(self._checkpoint_extra())
        path = os.path.join(self.out_dir, f"checkpoint_{self.tick}.pkl")
        with open(path + ".tmp", "wb") as f:  # write, then rename: a killed run never leaves a half-written save
            pickle.dump(data, f)
        os.replace(path + ".tmp", path)

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim = cls.__new__(cls)
        sim.size, sim.cfg, sim.dtype = d["size"], d["cfg"], d["w1"].dtype.type
        sim.H, sim.O = d["w1"].shape[2], d["w2"].shape[2]
        sim.checkpoint_every = CHECKPOINT_EVERY
        sim.use_numba = HAVE_NUMBA
        sim._complexity = (None, 0.0)
        sim.rng = np.random.default_rng()
        sim.rng.bit_generator.state = d["rng_state"]
        for k in ("food", "hazard", "cofood_grid", "cofood_age", "cofood_stats", "tick", "t", "next_id"):
            setattr(sim, k, d[k])
        for k in ("patch_xy", "patch_dir", "regrow_map"):
            if d.get(k) is not None:
                setattr(sim, k, d[k])
        sim.wall = d["wall"] if d.get("wall") is not None else np.zeros_like(sim.food)
        sim.shelter = d["shelter"] if d.get("shelter") is not None else np.zeros_like(sim.food)
        sim._has_shelter = bool(sim.shelter.any())
        n = len(d["x"])
        if "slot" in d:  # exact resume: same slots, same pool size
            sim._alloc(int(d["cap"]))
            pos = d["slot"]
        else:  # older saves: packed pool (resumes correctly, but not bit-for-bit)
            sim._alloc(max(256, int(n * 1.5)))
            pos = np.arange(n)
        for k in cls._FIELDS:
            if k != "alive" and k in d:
                getattr(sim, k)[pos] = d[k]
        sim.alive[pos] = True
        for k, v in d.get("meter", {}).items():
            setattr(sim, k, v)
        dd = np.arange(-SENSE_RADIUS, SENSE_RADIUS + 1)
        sim._ox, sim._oy = np.repeat(dd, 5), np.tile(dd, 5)
        sim.out_dir = out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            sim.log_path = os.path.join(out_dir, "log.csv")
        return sim

    def set_threads(self, n):
        """Threads for the compiled kernels (results do not depend on this: every
        agent's sensing and thinking is computed independently)."""
        if HAVE_NUMBA:
            import numba
            n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
            if n != numba.get_num_threads():
                numba.set_num_threads(n)

    def _sync_threads(self):
        # the batch runner (replicate_modes.py) writes the thread share for this run into
        # <out_dir>/THREADS and raises it as other runs finish
        try:
            with open(os.path.join(self.out_dir, "THREADS")) as f:
                self.set_threads(int(f.read().strip() or 1))
        except (OSError, ValueError):
            pass

    def run(self, ticks, extinction_stops=True, verbose=False):
        rows = []
        t0 = time.time()
        tick0 = self.tick
        if self.out_dir:
            self._births = []  # family tree: every birth with its parent and founding lineage
            self._sync_threads()
        for _ in range(ticks):
            self.step()
            if self.out_dir and self.tick % 500 == 0:
                self._sync_threads()
            if self.tick % LOG_EVERY == 0:
                rows.append(self.log_row())
                if self.out_dir and len(rows) >= 50:
                    self._flush(rows)
                    self._flush_births()
                    rows = []
            if self.out_dir and self.tick % 1000 == 0:
                self._novelty()
            if self.out_dir and self.tick % self.checkpoint_every == 0:
                self.checkpoint()
            if extinction_stops and self.population == 0:
                break
            if verbose and self.tick % 1000 == 0:
                print(f"tick {self.tick} pop {self.population} {(self.tick - tick0) / (time.time() - t0):.0f} ticks/s")
        if self.out_dir:
            self._flush(rows)
            self._flush_births()
            self.checkpoint()
        return rows

    def _flush(self, rows):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        if getattr(self, "_cofood_logged", None) != self.tick:  # cofood.csv in both arms (OFF logs spawns only)
            self._cofood_logged = self.tick  # once per tick (the final flush used to repeat the last row)
            path = os.path.join(self.out_dir, "cofood.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["tick"] + list(self.cofood_stats))
                w.writerow([self.tick] + list(self.cofood_stats.values()))


def founders_from_old_agents(agents):
    """Convert old-engine Agent objects (e.g. stage-13 founders) into FastSim arrays."""
    return dict(
        x=np.array([a.x for a in agents]), y=np.array([a.y for a in agents]),
        w1=np.stack([a.brain.w1 for a in agents]), b1=np.stack([a.brain.b1 for a in agents]),
        w2=np.stack([a.brain.w2 for a in agents]), b2=np.stack([a.brain.b2 for a in agents]),
        wh=np.stack([a.brain.wh for a in agents]),
        energy=np.array([a.energy for a in agents]),
    )


def from_old_checkpoint(path, stage13_dir):
    """Build a FastSim holding exactly the state of an old-engine (stage 13) checkpoint."""
    import sys
    sys.path.insert(0, stage13_dir)
    with open(path, "rb") as f:
        d = pickle.load(f)
    w, agents = d["world"], d["agents"]
    F = founders_from_old_agents(agents)
    sim = FastSim(seed=0, size=w.size, founders=F, social_sensing=w.social_sensing,
                  communication=w.communication, seasons=w.seasons, hazards=w.hazards,
                  cofood=w.cofood_on)
    n = len(agents)
    s = np.flatnonzero(sim.alive)[:n]
    sim.age[s] = [a.age for a in agents]
    sim.gen[s] = [a.generation for a in agents]
    sim.last_action[s] = [A.get(getattr(a, "last_action", None), -1) for a in agents]
    sim.food, sim.hazard = w.food.copy(), w.hazard.copy()
    sim.cofood_grid, sim.cofood_age = w.cofood.copy(), w.cofood_age.copy()
    sim.t, sim.tick = w.t, d["tick"]
    sim.rng = np.random.default_rng(d["tick"])
    return sim
