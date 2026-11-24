"""
Primordia GPU engine (PyTorch): the same world rules as engine.FastSim, with the
whole population held in tensors on the GPU (or CPU), so a tick is a handful of
batched tensor operations regardless of population size.

    sim = TorchSim(seed=1, device="cuda", **MODES["stage13"]["kw"](True))
    sim.run(25000)

Same rules as FastSim (see engine.py): costs, gains, food decay/regrowth,
seasons, hazards, signals, sharing, reproduction and mutation, shared food,
working memory, the sparse (patch) world. Checkpoints are written in FastSim's
format, so fast/evaluate_fast.py, lesion.py etc. read them unchanged.
Random numbers come from torch's generator, so runs match FastSim
statistically, not draw for draw.
"""
import csv
import os
import pickle
import time

import numpy as np
import torch

import engine as E

A = E.A


class TorchSim:
    def __init__(self, seed=1, size=48, out_dir=None, founders=None, init_population=80,
                 device=None, checkpoint_every=E.CHECKPOINT_EVERY, **kw):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dev = self.device
        # same defaults as FastSim
        cfg = dict(social_sensing=True, communication=True, seasons=True, hazards=True, cofood=False,
                   food_regrow_prob=0.02, food_decay_prob=0.01, max_food=1.0, season_period=600,
                   season_amplitude=0.9, hazard_fraction=0.03, cofood_spawn_prob=0.5, cofood_value=4.0,
                   cofood_life=150, hidden=E.HIDDEN_SIZE, n_actions=E.OUTPUT_SIZE,
                   init_inputs=E.INPUT_SIZE, memory=False, language=False, food_mode="uniform", n_patches=16,
                   patch_sigma=4.0, patch_speed=0.03, patch_turn=0.05, patch_peak=0.057, patch_life=0.0)
        near = kw.pop("founders_near_patches", False)
        cfg.update(kw)
        self.cfg, self.size = cfg, size
        self.H, self.O = int(cfg["hidden"]), int(cfg["n_actions"])
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.np_rng = np.random.default_rng(seed + 10_000)  # small CPU-side choices
        self.checkpoint_every = checkpoint_every
        S = size
        self.food = torch.rand((S, S), generator=self.g, device=dev) * cfg["max_food"]
        self.food *= (torch.rand((S, S), generator=self.g, device=dev) < 0.3)
        self.hazard = torch.zeros((S, S), device=dev)
        self.cofood_grid = torch.zeros((S, S), device=dev)
        self.cofood_age = torch.zeros((S, S), device=dev)
        self.cofood_stats = {"spawned": 0, "eaten": 0, "expired": 0, "solo_attempts": 0, "co_eaters": 0}
        self.t = 0
        self.tick = 0
        self.next_id = 0
        self._complexity = (None, 0.0)
        d = torch.arange(-E.SENSE_RADIUS, E.SENSE_RADIUS + 1, device=dev)
        self._ox = d.repeat_interleave(5)
        self._oy = d.repeat(5)
        if cfg["food_mode"] == "patches":
            K = cfg["n_patches"]
            self.patch_xy = self.np_rng.uniform(0, S, (K, 2))
            self.patch_dir = self.np_rng.uniform(0, 2 * np.pi, K)
            self._update_patch_map()
            self.food = cfg["max_food"] * (torch.rand((S, S), generator=self.g, device=dev)
                                          < self.regrow_map / cfg["patch_peak"] * 0.5).float()
        self._alloc(256)
        if founders is not None:
            f = dict(founders)
            if near and cfg["food_mode"] == "patches":
                k = self.np_rng.integers(0, cfg["n_patches"], len(f["x"]))
                f["x"] = np.round(self.patch_xy[k, 0] + self.np_rng.normal(0, cfg["patch_sigma"], len(k))).astype(int) % S
                f["y"] = np.round(self.patch_xy[k, 1] + self.np_rng.normal(0, cfg["patch_sigma"], len(k))).astype(int) % S
            self.add_agents(**f)
        else:
            n, H, O = init_population, self.H, self.O
            r = self.np_rng
            self.add_agents(x=r.integers(0, S, n), y=r.integers(0, S, n),
                            w1=r.normal(0, np.sqrt(2.0 / cfg["init_inputs"]), (n, E.INPUT_SIZE, H)),
                            b1=np.zeros((n, H)), w2=r.normal(0, np.sqrt(2.0 / H), (n, H, O)),
                            b2=np.zeros((n, O)), wh=r.normal(0, 0.1 / np.sqrt(H), (n, H, H)))
        self.out_dir = out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self.log_path = os.path.join(out_dir, "log.csv")
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", newline="") as fh:
                    csv.writer(fh).writerow(["tick", "population", "mean_energy", "mean_age",
                                             "max_generation", "mean_complexity", "total_food"])

    # ------------------------------------------------------------------ pool
    _FIELDS = ["alive", "x", "y", "energy", "age", "gen", "aid", "parent", "last_action",
               "w1", "b1", "w2", "b2", "wh", "hstate", "ws", "bs", "last_sym2", "lineage"]

    def _alloc(self, cap):
        dev, H, O = self.device, self.H, self.O
        self.cap = cap
        self.alive = torch.zeros(cap, dtype=torch.bool, device=dev)
        for k in ("x", "y", "age", "gen"):
            setattr(self, k, torch.zeros(cap, dtype=torch.long, device=dev))
        self.aid = torch.full((cap,), -1, dtype=torch.long, device=dev)
        self.parent = torch.full((cap,), -1, dtype=torch.long, device=dev)
        self.last_action = torch.full((cap,), -1, dtype=torch.long, device=dev)
        self.energy = torch.zeros(cap, dtype=torch.float64, device=dev)
        self.w1 = torch.zeros((cap, E.INPUT_SIZE, H), device=dev)
        self.b1 = torch.zeros((cap, H), device=dev)
        self.w2 = torch.zeros((cap, H, O), device=dev)
        self.b2 = torch.zeros((cap, O), device=dev)
        self.wh = torch.zeros((cap, H, H), device=dev)
        self.hstate = torch.zeros((cap, H), device=dev)
        self.ws = torch.zeros((cap, H, E.N_SYMBOLS), device=dev)  # second-symbol head (language)
        self.bs = torch.zeros((cap, E.N_SYMBOLS), device=dev)
        self.last_sym2 = torch.zeros(cap, dtype=torch.long, device=dev)
        self.lineage = torch.full((cap,), -1, dtype=torch.long, device=dev)

    def _grow(self, need):
        old = {k: getattr(self, k) for k in self._FIELDS}
        self._alloc(max(need, int(self.cap * 1.5) + 64))
        for k, v in old.items():
            getattr(self, k)[:len(v)] = v

    def _compact(self):
        idx = self.alive.nonzero().squeeze(1)
        for k in self._FIELDS:
            arr = getattr(self, k)
            arr[:len(idx)] = arr[idx].clone()
        self.alive[:] = False
        self.alive[:len(idx)] = True

    def _t(self, a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

    def add_agents(self, x, y, w1, b1, w2, b2, wh, energy=None, gen=None, parent=None, ws=None, bs=None,
                   lineage=None):
        n = len(x)
        free = (~self.alive).nonzero().squeeze(1)
        if len(free) < n:
            self._grow(self.cap + n - len(free))
            free = (~self.alive).nonzero().squeeze(1)
        s = free[:n]
        L = torch.long
        self.alive[s] = True
        self.x[s] = self._t(x, L) % self.size if not torch.is_tensor(x) else x % self.size
        self.y[s] = self._t(y, L) % self.size if not torch.is_tensor(y) else y % self.size
        self.energy[s] = E.START_ENERGY if energy is None else (energy if torch.is_tensor(energy) else self._t(energy, torch.float64))
        self.age[s] = 0
        self.gen[s] = 0 if gen is None else (gen if torch.is_tensor(gen) else self._t(gen, L))
        self.parent[s] = -1 if parent is None else (parent if torch.is_tensor(parent) else self._t(parent, L))
        self.aid[s] = torch.arange(self.next_id, self.next_id + n, device=self.device)
        self.lineage[s] = self.aid[s] if lineage is None else lineage
        if getattr(self, "_births", None) is not None and parent is not None:
            self._births.append(torch.stack([torch.full((n,), self.tick + 1, device=self.device), self.aid[s],
                                             parent, gen, self.lineage[s]], 1).cpu().numpy())
        self.next_id += n
        self.last_action[s] = -1
        for k, v in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2), ("wh", wh)):
            getattr(self, k)[s] = v if torch.is_tensor(v) else self._t(v)
        self.hstate[s] = 0.0
        self.ws[s] = ws if ws is not None else torch.randn((n, self.H, E.N_SYMBOLS), generator=self.g, device=self.device) * 0.05
        self.bs[s] = bs if bs is not None else 0.0
        self.last_sym2[s] = 0

    @property
    def population(self):
        return int(self.alive.sum())

    # ------------------------------------------------------------------ world
    def _rand(self, *shape):
        return torch.rand(shape, generator=self.g, device=self.device)

    def _update_patch_map(self):
        c, S = self.cfg, self.size
        g = np.arange(S)
        sig2 = 2 * c["patch_sigma"] ** 2
        def bumps(centres):
            d = np.abs(g[None, :] - centres[:, None]); d = np.minimum(d, S - d)
            return np.exp(-d ** 2 / sig2)
        m = np.minimum(c["patch_peak"] * np.einsum("kx,ky->xy", bumps(self.patch_xy[:, 0]),
                                                   bumps(self.patch_xy[:, 1])), 1.0)
        self.regrow_map_np = m
        self._map_cum = np.cumsum(m.ravel())
        self.regrow_map = torch.as_tensor(m, dtype=torch.float32, device=self.device)

    def _sample_patch_cell(self):
        k = int(np.searchsorted(self._map_cum, self.np_rng.random() * self._map_cum[-1]))
        return k // self.size, k % self.size

    def _world_step(self):
        c, S = self.cfg, self.size
        self.food[self._rand(S, S) < c["food_decay_prob"]] *= 0.5
        self.t += 1
        if c["hazards"] and (self.t == 1 or self.t % c["season_period"] == 0):
            self.hazard = (self._rand(S, S) < c["hazard_fraction"]).float()
        self._cofood_step()
        p = c["food_regrow_prob"]
        if c["seasons"]:
            p = p * (1 + c["season_amplitude"] * np.sin(2 * np.pi * self.t / c["season_period"]))
        rr = self._rand(S, S)
        if c["food_mode"] == "patches":
            self.patch_dir += self.np_rng.normal(0, c["patch_turn"], len(self.patch_dir))
            self.patch_xy[:, 0] = (self.patch_xy[:, 0] + c["patch_speed"] * np.cos(self.patch_dir)) % S
            self.patch_xy[:, 1] = (self.patch_xy[:, 1] + c["patch_speed"] * np.sin(self.patch_dir)) % S
            moved = False
            if c["patch_life"] > 0:
                gone = self.np_rng.random(len(self.patch_dir)) < 1.0 / c["patch_life"]
                if gone.any():
                    self.patch_xy[gone] = self.np_rng.uniform(0, S, (int(gone.sum()), 2))
                    moved = True
            if self.t % 10 == 0 or moved:
                self._update_patch_map()
            thresh = self.regrow_map * (p / c["food_regrow_prob"])
        else:
            thresh = p
        self.food[(self.food < 0.05) & (rr < thresh)] = c["max_food"]
        self.food.clamp_(0, c["max_food"])

    def _cofood_step(self):
        c, S = self.cfg, self.size
        if c["cofood_spawn_prob"] <= 0:
            return
        if c["cofood"]:
            live = self.cofood_grid > 0
            self.cofood_age[live] += 1
            old = live & (self.cofood_age > c["cofood_life"])
            self.cofood_stats["expired"] += int(old.sum())
            self.cofood_grid[old] = 0
        if self.np_rng.random() < c["cofood_spawn_prob"]:
            self.cofood_stats["spawned"] += 1
            pick = self._sample_patch_cell if c["food_mode"] == "patches" else \
                (lambda: (int(self.np_rng.integers(S)), int(self.np_rng.integers(S))))
            if c["cofood"]:
                x, y = pick()
                self.cofood_grid[x, y] += c["cofood_value"]
                self.cofood_age[x, y] = 0
            else:
                left = c["cofood_value"]
                while left > 1e-9:
                    cx, cy = pick()
                    add = min(c["max_food"] - float(self.food[cx, cy]), left)
                    self.food[cx, cy] += add
                    left -= add

    # ------------------------------------------------------------------ tick
    def step(self):
        c, S, dev = self.cfg, self.size, self.device
        self._world_step()
        idx = self.alive.nonzero().squeeze(1)
        n = len(idx)
        if n and n < 0.7 * (int(idx[-1]) + 1) and int(idx[-1]) > 256:
            self._compact()
            idx = self.alive.nonzero().squeeze(1)
        if n == 0:
            self.tick += 1
            return
        hi = int(idx[-1]) + 1
        x0, y0 = self.x[idx], self.y[idx]
        cell0 = x0 * S + y0
        # occupancy and last tick's signals (before anyone acts)
        occ = torch.zeros(S * S, device=dev).index_add_(0, cell0, torch.ones(n, device=dev))
        sig = torch.zeros(S * S, device=dev)
        if c["communication"]:
            la = self.last_action[idx]
            s = (la >= E.SIG0) & (la < E.SIG0 + E.N_SYMBOLS)
            if s.any():
                sig.scatter_reduce_(0, cell0[s], (la[s] - E.SIG0 + 1).float(), reduce="amax")
        sig2 = torch.zeros(S * S, device=dev)
        if c["communication"] and c["language"]:
            la = self.last_action[idx]
            s = (la >= E.SIG0) & (la < E.SIG0 + E.N_SYMBOLS)
            if s.any():
                sig2.scatter_reduce_(0, cell0[s], self.last_sym2[idx][s].float(), reduce="amax")
        self.age[idx] += 1
        # sensing: 5x5 patches, stage-13 input layout
        cells = ((x0[:, None] + self._ox) % S) * S + (y0[:, None] + self._oy) % S
        X = torch.zeros((n, E.INPUT_SIZE), device=dev)
        X[:, E.P_FOOD:E.P_FOOD + 25] = self.food.view(-1)[cells]
        if c["social_sensing"]:
            o = occ[cells]
            o[:, 12] = (o[:, 12] - 1).clamp(min=0)
            X[:, E.P_OCC:E.P_OCC + 25] = o
        X[:, E.P_SIG:E.P_SIG + 25] = sig[cells]
        X[:, E.P_HAZ:E.P_HAZ + 25] = self.hazard.view(-1)[cells]
        X[:, E.P_COFOOD:E.P_COFOOD + 25] = (self.cofood_grid.view(-1)[cells] > 0).float()
        X[:, E.P_SIG2:E.P_SIG2 + 25] = sig2[cells]
        X[:, E.P_EXTRA] = (self.energy[idx] / E.MAX_ENERGY).float()
        X[:, E.P_EXTRA + 1] = self.age[idx].float() / E.MAX_AGE
        # brains, batched over slots [0, hi) (dead slots are computed and ignored)
        Xf = torch.zeros((hi, E.INPUT_SIZE), device=dev)
        Xf[idx] = X
        pre = torch.bmm(Xf.unsqueeze(1), self.w1[:hi]).squeeze(1) + self.b1[:hi]
        if c["memory"]:
            pre += torch.bmm(self.hstate[:hi].unsqueeze(1), self.wh[:hi]).squeeze(1)
        h = torch.tanh(pre)
        if c["memory"]:
            self.hstate[idx] = h[idx]
        out = (torch.bmm(h.unsqueeze(1), self.w2[:hi]).squeeze(1) + self.b2[:hi])[idx]
        p = torch.softmax(out.double(), dim=1)
        u = torch.rand(n, generator=self.g, device=dev, dtype=torch.float64)
        act = (torch.cumsum(p, 1) < u[:, None]).sum(1).clamp(max=self.O - 1)
        if getattr(self, "_act_counts", None) is None:
            self._act_counts = torch.zeros(E.OUTPUT_SIZE, dtype=torch.long, device=dev)
        self._act_counts += torch.bincount(act, minlength=E.OUTPUT_SIZE)[:E.OUTPUT_SIZE]
        # second symbol of each message (chosen whenever the agent signals)
        u2 = torch.rand(n, generator=self.g, device=dev, dtype=torch.float64)
        sym2 = torch.zeros(n, dtype=torch.long, device=dev)
        sg = ((act >= E.SIG0) & (act < E.SIG0 + E.N_SYMBOLS)).nonzero().squeeze(1)
        if len(sg):
            hs = h[idx[sg]]
            lg = torch.bmm(hs.unsqueeze(1), self.ws[idx[sg]]).squeeze(1) + self.bs[idx[sg]]
            pq = torch.softmax(lg.double(), 1)
            sym2[sg] = (torch.cumsum(pq, 1) < u2[sg, None]).sum(1).clamp(max=E.N_SYMBOLS - 1) + 1
        self.last_sym2[idx] = sym2
        keys = torch.rand(n, generator=self.g, device=dev, dtype=torch.float64)
        su = torch.rand(n, generator=self.g, device=dev, dtype=torch.float64)
        # actions
        e = self.energy[idx].clone()
        idle = (act == A["stay"]) | (act == A["eat"]) | (act == A["reproduce"]) | (act == A["mate"])
        e[idle] -= E.IDLE_COST
        dx = torch.where(act == A["E"], 1, torch.where(act == A["W"], -1, 0))
        dy = torch.where(act == A["S"], 1, torch.where(act == A["N"], -1, 0))
        x1, y1 = (x0 + dx) % S, (y0 + dy) % S
        e[(act >= A["N"]) & (act <= A["W"])] -= E.MOVE_COST
        e[(act >= E.SIG0) & (act < E.SIG0 + E.N_SYMBOLS)] -= E.SIGNAL_COST
        e[act == A["mark"]] -= E.MARK_COST
        # eating: within a cell, in random-key order, each takes up to EAT_GAIN
        eat = (act == A["eat"]).nonzero().squeeze(1)
        if len(eat):
            ec = cell0[eat]
            order = torch.argsort(ec.double() + keys[eat])
            cs = ec[order]
            first = torch.ones_like(cs, dtype=torch.bool)
            first[1:] = cs[1:] != cs[:-1]
            pos = torch.arange(len(cs), device=dev)
            start = torch.cummax(torch.where(first, pos, torch.zeros_like(pos)), 0).values
            rank = (pos - start).double()
            gain_s = (self.food.view(-1)[cs].double() - E.EAT_GAIN * rank).clamp(0, E.EAT_GAIN)
            g = torch.empty_like(gain_s)
            g[order] = gain_s
            e[eat] += g
            self.food.view(-1).index_add_(0, ec, -g.float())
            self.food.clamp_(min=0)
        rep = ((act == A["reproduce"]) & (e >= E.REPRODUCE_MIN_ENERGY)).nonzero().squeeze(1)
        e[rep] -= E.REPRODUCE_COST
        # sharing (few agents): candidates in the 3x3 around start positions, done on the CPU
        sh = (act == A["share"]).nonzero().squeeze(1)
        recv = None
        if len(sh):
            e[sh] -= E.SHARE_COST
            recv = self._share(sh, cell0, x0, y0, e, su)
        # poison, cap, deaths
        e[self.hazard[x1, y1] > 0] -= E.HAZARD_DRAIN
        e.clamp_(max=E.MAX_ENERGY)
        dead = (e <= 0) | (self.age[idx] >= E.MAX_AGE)
        self.x[idx], self.y[idx] = x1, y1
        self.energy[idx] = e
        self.last_action[idx] = act
        self.alive[idx[dead]] = False
        if recv is not None:
            rc = idx[recv]
            rc = rc[self.alive[rc]]
            self.energy.index_add_(0, rc, torch.full((len(rc),), E.SHARE_AMOUNT * E.SHARE_MULTIPLIER,
                                                     dtype=torch.float64, device=dev))
            self.energy.clamp_(max=E.MAX_ENERGY)
        if c["cofood"] and len(eat):
            self._resolve_cofood(idx[eat], cell0[eat])
        if len(rep):
            self._spawn(idx[rep])
        self.tick += 1

    def _share(self, sh, cell0, x0, y0, e, su):
        """Givers with energy above the gift pick a random other agent within radius 1."""
        S = self.size
        c0, xs, ys = cell0.cpu().numpy(), x0.cpu().numpy(), y0.cpu().numpy()
        ee, uu = e.cpu().numpy(), su.cpu().numpy()
        order = np.argsort(c0, kind="stable")
        cs = c0[order]
        givers, recv = [], []
        for j in sh.cpu().numpy():
            if ee[j] <= E.SHARE_AMOUNT:
                continue
            cands = []
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    cc = ((xs[j] + ddx) % S) * S + (ys[j] + ddy) % S
                    lo, hi = np.searchsorted(cs, cc), np.searchsorted(cs, cc, side="right")
                    cands.extend(order[lo:hi].tolist())
            cands = [k for k in cands if k != j]
            if cands:
                givers.append(j)
                recv.append(cands[int(uu[j] * len(cands))])
        if not givers:
            return None
        gi = torch.as_tensor(givers, device=self.device)
        e[gi] -= E.SHARE_AMOUNT
        return torch.as_tensor(recv, device=self.device)

    def _resolve_cofood(self, eaters, ecell):
        has = self.cofood_grid.view(-1)[ecell] > 0
        if not has.any():
            return
        eaters, ecell = eaters[has], ecell[has]
        ok = self.alive[eaters]
        el, cl = eaters[ok].cpu().numpy(), ecell[ok].cpu().numpy()
        cells, counts = np.unique(cl, return_counts=True)
        for cc, m in zip(cells, counts):
            if m >= 2:
                sel = torch.as_tensor(el[cl == cc], device=self.device)
                share = float(self.cofood_grid.view(-1)[cc]) / m
                self.energy[sel] = (self.energy[sel] + share).clamp(max=E.MAX_ENERGY)
                self.cofood_grid.view(-1)[cc] = 0
                self.cofood_stats["eaten"] += 1
                self.cofood_stats["co_eaters"] += int(m)
            else:
                self.cofood_stats["solo_attempts"] += int(m)

    def _spawn(self, par):
        def mut(a):
            a = a.clone()
            m = torch.rand(a.shape, generator=self.g, device=self.device) < E.MUT_RATE
            a += m * torch.randn(a.shape, generator=self.g, device=self.device) * E.MUT_STRENGTH
            return a
        self.add_agents(self.x[par].clone(), self.y[par].clone(), mut(self.w1[par]), mut(self.b1[par]),
                        mut(self.w2[par]), mut(self.b2[par]), mut(self.wh[par]),
                        ws=mut(self.ws[par]), bs=mut(self.bs[par]),
                        energy=torch.full((len(par),), E.REPRODUCE_COST * 0.5, dtype=torch.float64, device=self.device),
                        gen=self.gen[par] + 1, parent=self.aid[par], lineage=self.lineage[par])

    # ------------------------------------------------------------------ io
    def log_row(self):
        idx = self.alive.nonzero().squeeze(1)
        if not len(idx):
            return [self.tick, 0, 0, 0, 0, 0, float(self.food.sum())]
        if self._complexity[0] is None or self.tick - self._complexity[0] >= 500:
            smp = idx[:: max(1, len(idx) // 200)]
            comp = self.w1[smp].abs().mean((1, 2)) + self.w2[smp].abs().mean((1, 2))
            self._complexity = (self.tick, float(comp.mean()))
        return [self.tick, len(idx), float(self.energy[idx].mean()), float(self.age[idx].float().mean()),
                int(self.gen[idx].max()), self._complexity[1], float(self.food.sum())]

    def checkpoint(self):
        """Write a checkpoint in FastSim's format (readable by engine.FastSim.from_checkpoint)."""
        if not self.out_dir:
            return
        idx = self.alive.nonzero().squeeze(1)
        cpu = lambda t: t[idx].cpu().numpy()
        d = {k: cpu(getattr(self, k)) for k in self._FIELDS if k != "alive"}
        d["energy"] = d["energy"].astype(np.float64)
        d.update(tick=self.tick, t=self.t, next_id=self.next_id, cfg=dict(self.cfg), size=self.size,
                 food=self.food.double().cpu().numpy(), hazard=self.hazard.double().cpu().numpy(),
                 cofood_grid=self.cofood_grid.double().cpu().numpy(),
                 cofood_age=self.cofood_age.double().cpu().numpy(), cofood_stats=dict(self.cofood_stats),
                 rng_state=np.random.default_rng(self.tick).bit_generator.state, engine="torch-v1",
                 patch_xy=getattr(self, "patch_xy", None), patch_dir=getattr(self, "patch_dir", None),
                 regrow_map=getattr(self, "regrow_map_np", None))
        path = os.path.join(self.out_dir, f"checkpoint_{self.tick}.pkl")
        with open(path + ".tmp", "wb") as f:
            pickle.dump(d, f)
        os.replace(path + ".tmp", path)

    def _novelty(self):
        cnt = getattr(self, "_act_counts", None)
        if cnt is None or int(cnt.sum()) == 0:
            return
        share = (cnt.double() / cnt.sum()).cpu().numpy()
        prev = getattr(self, "_prev_share", None)
        change = "" if prev is None else round(float(np.abs(share - prev).sum()), 4)
        idx = self.alive.nonzero().squeeze(1)
        lineages = int(torch.unique(self.lineage[idx]).numel()) if len(idx) else 0
        smp = idx[:: max(1, len(idx) // 200)]
        spread = float(self.w2[smp].reshape(len(smp), -1).std(0).mean()) if len(smp) > 1 else 0.0
        path = os.path.join(self.out_dir, "novelty.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick", "change_from_previous_window", "lineages_alive", "weight_spread"] + E.ACTIONS)
            w.writerow([self.tick, change, lineages, round(spread, 4)] + [round(float(x), 4) for x in share])
        self._prev_share = share
        self._act_counts.zero_()

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

    def run(self, ticks, extinction_stops=True, verbose=False):
        rows = []
        t0 = time.time()
        if self.out_dir:
            self._births = []
        for _ in range(ticks):
            self.step()
            if self.tick % E.LOG_EVERY == 0:
                rows.append(self.log_row())
                if self.out_dir and len(rows) >= 50:
                    self._flush(rows); self._flush_births(); rows = []
            if self.out_dir and self.tick % 1000 == 0:
                self._novelty()
            if self.out_dir and self.tick % self.checkpoint_every == 0:
                self.checkpoint()
            if extinction_stops and self.population == 0:
                break
            if verbose and self.tick % 1000 == 0:
                print(f"tick {self.tick} pop {self.population} {self.tick / (time.time() - t0):.0f} ticks/s", flush=True)
        if self.out_dir:
            self._flush(rows)
            self._flush_births()
            self.checkpoint()
        return rows

    def _flush(self, rows):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        path = os.path.join(self.out_dir, "cofood.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tick"] + list(self.cofood_stats))
            w.writerow([self.tick] + list(self.cofood_stats.values()))
