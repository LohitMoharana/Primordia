"""
Survival test for fast-engine checkpoints, the same test as lab/evaluate.py:
sample 50 agents, run each ALONE for 150 steps in its own copy of the world
(the rest of the population frozen where it was at the checkpoint, as in the
old test), and report eat-fraction, energy change, steps survived and the
share whose energy rose. All 50 run at once.

    python3 evaluate_fast.py <on_ckpt> <off_ckpt>        # ON-vs-OFF table
    python3 evaluate_fast.py <ckpt>                       # one profile
"""
import pickle
import sys

import numpy as np

import engine as E
from engine import FastSim

N, STEPS, SEED = 50, 150, 0


def profile(path, n=N, steps=STEPS, seed=SEED, world_path=None, reset_state=False, override=None):
    """world_path: test the agents of `path` in the world of another checkpoint
    (food, poison, shared food, clock, world rules and the frozen population
    there), so two arms' agents can be compared in the same world."""
    sim = FastSim.from_checkpoint(path)
    if world_path:
        w = FastSim.from_checkpoint(world_path)
        sim.food, sim.hazard, sim.cofood_grid, sim.cofood_age = w.food, w.hazard, w.cofood_grid, w.cofood_age
        sim.t, sim.size = w.t, w.size
        sim.wall = w.wall
        keep = {k: sim.cfg[k] for k in ("hidden", "n_actions", "init_inputs", "memory") if k in sim.cfg}
        sim.cfg = dict(w.cfg, **keep)
        sim.rng = np.random.default_rng(seed + 1)
        for k in ("patch_xy", "patch_dir", "regrow_map"):
            if hasattr(w, k):
                setattr(sim, k, getattr(w, k))
        sim._frozen = (w.x[w.alive].copy(), w.y[w.alive].copy(), w.last_action[w.alive].copy())
        sim._frozen_sym2 = w.last_sym2[w.alive].copy()
    if override:  # e.g. {"social_sensing": False}: same brains, one channel switched off (lesion test)
        sim.cfg = dict(sim.cfg, **override)
    S = sim.size
    rng = np.random.default_rng(seed)
    alive = np.flatnonzero(sim.alive)
    pick = rng.choice(alive, size=min(n, len(alive)), replace=False)
    k = len(pick)
    # frozen population: occupancy and last tick's signals, as at the checkpoint
    fx, fy, fl = getattr(sim, "_frozen", (sim.x[alive], sim.y[alive], sim.last_action[alive]))
    fs2 = sim.last_sym2[alive] if not hasattr(sim, "_frozen") else getattr(sim, "_frozen_sym2", np.zeros(len(fx), np.int64))
    occ = np.zeros((S, S))
    np.add.at(occ, (fx, fy), 1)
    sig = np.zeros((S, S))
    if sim.cfg["communication"]:
        s = (fl >= E.SIG0) & (fl < E.SIG0 + E.N_SYMBOLS)
        sig[fx[s], fy[s]] = fl[s] - E.SIG0 + 1
    sig2 = np.zeros((S, S))  # second symbols of the frozen population's last messages (language)
    if sim.cfg["communication"] and sim.cfg.get("language", False):
        s = (fl >= E.SIG0) & (fl < E.SIG0 + E.N_SYMBOLS)
        sig2[fx[s], fy[s]] = fs2[s]
    # each tested agent gets its own food grid; world randomness is shared,
    # as it was in the old test (every copy started from the same world RNG state)
    foods = np.repeat(sim.food[None], k, axis=0)
    x, y = sim.x[pick].copy(), sim.y[pick].copy()
    e = sim.energy[pick].copy()
    age = sim.age[pick].copy()
    if reset_state:  # same starting energy and age for everyone: compares the brains only
        e[:] = 5.0
        age[:] = 100
    start = e.copy()
    alive_t = np.ones(k, bool)
    survived = np.zeros(k, int)
    eats = np.zeros(k)
    hmem = np.zeros((k, sim.H), np.float32)  # memory starts empty for the test
    on_poison = 0.0
    cover = []
    acts = np.zeros(sim.O)
    for _ in range(steps):
        sim._world_step(foods)  # same decay/regrowth draws for every private grid
        live = np.flatnonzero(alive_t)
        if not len(live):
            break
        age[live] += 1
        flat = ((x[live, None] + sim._ox) % S) * S + (y[live, None] + sim._oy) % S
        X = np.zeros((len(live), E.INPUT_SIZE), sim.dtype)
        X[:, :25] = foods[live].reshape(len(live), -1)[np.arange(len(live))[:, None], flat]
        if sim.cfg["social_sensing"]:
            o = occ.ravel()[flat].copy()
            o[:, 12] = np.maximum(0, o[:, 12] - 1)
            X[:, 25:50] = o
        X[:, 50:75] = sig.ravel()[flat]
        X[:, 75:100] = sim.hazard.ravel()[flat]
        X[:, 200:225] = sim.cofood_grid.ravel()[flat] > 0
        X[:, 175:200] = sig2.ravel()[flat]
        if sim.wall.any():
            X[:, 125:150] = sim.wall.ravel()[flat] > 0
        X[:, 225] = e[live] / E.MAX_ENERGY
        X[:, 226] = age[live] / E.MAX_AGE
        slots = pick[live]
        pre = np.einsum("ni,nih->nh", X, sim.w1[slots]) + sim.b1[slots]
        if sim.cfg.get("memory", False):  # working memory carried across the test steps
            pre += np.einsum("ni,nij->nj", hmem[live], sim.wh[slots])
        h = np.tanh(pre)
        if sim.cfg.get("memory", False):
            hmem[live] = h
        out = np.einsum("nh,nho->no", h, sim.w2[slots]) + sim.b2[slots]
        p = np.exp((out - out.max(1, keepdims=True)).astype(float))
        p /= p.sum(1, keepdims=True)
        act = np.minimum((np.cumsum(p, 1) < rng.random(len(live))[:, None]).sum(1), sim.O - 1)
        np.add.at(acts, act, 1)
        ee = e[live]
        ee[np.isin(act, [E.A["stay"], E.A["eat"], E.A["reproduce"], E.A["mate"]])] -= E.IDLE_COST
        mv = (act >= E.A["N"]) & (act <= E.A["W"])
        x0, y0 = x[live], y[live]
        dx = np.select([act == E.A["E"], act == E.A["W"]], [1, -1], 0)
        dy = np.select([act == E.A["S"], act == E.A["N"]], [1, -1], 0)
        ee[mv] -= E.MOVE_COST
        ee[(act >= E.SIG0) & (act < E.SIG0 + E.N_SYMBOLS)] -= E.SIGNAL_COST
        ee[act == E.A["mark"]] -= E.MARK_COST
        et = act == E.A["eat"]
        if et.any():
            li = live[et]
            g = np.minimum(foods[li, x0[et], y0[et]], E.EAT_GAIN)
            foods[li, x0[et], y0[et]] -= g
            ee[et] += g
            eats[li] += 1
        rep = (act == E.A["reproduce"]) & (ee >= E.REPRODUCE_MIN_ENERGY)
        ee[rep] -= E.REPRODUCE_COST
        sh = act == E.A["share"]
        if sh.any():
            ee[sh] -= E.SHARE_COST
            nb = np.zeros(sh.sum())
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    nb += occ[(x0[sh] + ddx) % S, (y0[sh] + ddy) % S]
            give = (nb > 0) & (ee[sh] > E.SHARE_AMOUNT)
            tmp = ee[sh]
            tmp[give] -= E.SHARE_AMOUNT
            ee[sh] = tmp
        nx, ny = (x0 + dx) % S, (y0 + dy) % S
        blocked = sim.wall[nx, ny] > 0
        x[live], y[live] = np.where(blocked, x0, nx), np.where(blocked, y0, ny)
        hz = sim.hazard[x[live], y[live]] > 0
        on_poison += hz.sum()
        cover.append(sim.hazard.mean())
        ee[hz] -= E.HAZARD_DRAIN
        np.minimum(ee, E.MAX_ENERGY, out=ee)
        e[live] = ee
        survived[live] += 1
        dead = (ee <= 0) | (age[live] >= E.MAX_AGE)
        alive_t[live[dead]] = False
    tot = acts.sum() or 1
    de = e - start
    return {
        "population": len(alive), "gen": int(sim.gen[alive].max()),
        "eat": float(np.mean(eats / np.maximum(survived, 1))),
        "energy": float(de.mean()), "survival": float(survived.mean()),
        "improving": float((de > 0).mean()),
        # poison avoidance: share of agent-ticks on poison vs share of cells that are poison
        "poison_ratio": float((on_poison / max(survived.sum(), 1)) / np.mean(cover)) if cover and np.mean(cover) > 0 else float("nan"),
        "action_share": {E.ACTIONS[i]: acts[i] / tot for i in range(sim.O) if acts[i]},
    }


def table(on, off):
    rows = [("eat-fraction", "eat"), ("energy change", "energy"), ("steps survived /150", "survival"),
            ("share improving", "improving"), ("population", "population")]
    print(f"{'metric':<22}{'OFF':>10}{'ON':>10}{'ON-OFF':>10}")
    for label, k in rows:
        print(f"{label:<22}{off[k]:>10.3f}{on[k]:>10.3f}{on[k] - off[k]:>+10.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        table(profile(sys.argv[1]), profile(sys.argv[2]))
    else:
        print(profile(sys.argv[1]))
