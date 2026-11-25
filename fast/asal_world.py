"""
ASAL-style search on Primordia, part 1: the world side.

    decode(u)        -> FastSim keyword arguments for a point u in [0, 1]^D
    rollout(job)     -> run one world and return its rendered frames (worker process)
    render(sim)      -> one RGB picture of the world (uint8, 256 x 256)

The search space covers what kind of world the creatures live in (how food is
spread and moves, seasons, poison, buried food, predators, shelters), never
what the creatures do: brains still start as the same 300 evolved stage-5
founders and change only by mutation and selection.

Pictures (render): two views of the creatures only, never the food or soil.
"space": where creatures (coloured by their genes) and predators (red) stand;
"genes": creatures placed by their genes. Colours and gene axes are fixed random
projections of each creature's output weights and gut, scaled by the founders,
identical in every world, so evolution shows as change in colour and position.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE13 = os.path.join(HERE, "..", "stages", "13_cofood")

# name, low, high, scale ("lin", "log", "int")
SPACE = [
    ("n_patches", 2, 24, "int"),        # how many food clusters
    ("richness", 0.4, 2.0, "log"),      # total surface food relative to sparse world v2
    ("patch_speed", 0.0, 0.12, "lin"),  # how fast clusters drift
    ("patch_life", 100, 4000, "log"),   # mean life of a cluster (top 10% of the range = never vanish)
    ("season_amp", 0.0, 0.95, "lin"),   # seasons off below 0.05
    ("season_period", 200, 3000, "log"),
    ("hazard_frac", 0.0, 0.08, "lin"),  # poison off below 0.005
    ("n_soil", 0, 10, "int"),           # buried-food patches (0 = no buried food)
    ("buried_base", 0.0, 1.0, "lin"),   # how well a surface gut digests buried food
    ("n_pred", 0, 80, "int"),           # predators (off below 5)
    ("catch_p", 0.05, 0.4, "log"),
    ("kill_gain", 1.0, 3.0, "lin"),
    ("shelter_frac", 0.0, 0.05, "lin"), # off below 0.003
    ("gut_q", 1.0, 2.5, "lin"),         # how costly being a generalist is
]
D = len(SPACE)
NAMES = [s[0] for s in SPACE]


def default_start():
    """A world known to stay healthy (1000-2300 creatures in tests): no predators, no poison,
    no seasons, some buried food, surface food a little below sparse world v2. Searches start
    here (as ASAL starts from a substrate's default settings) and move outward."""
    u = np.full(D, 0.5)
    for k in ("n_pred", "hazard_frac", "season_amp", "shelter_frac"):
        u[NAMES.index(k)] = 0.0
    u[NAMES.index("richness")] = 0.8
    u[NAMES.index("n_soil")] = 0.3
    return u


def raw_params(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    out = {}
    for (name, lo, hi, kind), v in zip(SPACE, u):
        if kind == "log":
            out[name] = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
        elif kind == "int":
            out[name] = int(round(lo + v * (hi - lo)))
        else:
            out[name] = float(lo + v * (hi - lo))
    out["_patch_forever"] = bool(u[NAMES.index("patch_life")] > 0.9)
    return out


def decode(u):
    """Point in [0,1]^D -> keyword arguments for WaysSim (sparse world v2 as the base)."""
    p = raw_params(u)
    n = max(2, p["n_patches"])
    kw = dict(size=128, food_mode="patches", n_patches=n, patch_sigma=4.0,
              patch_speed=p["patch_speed"], patch_turn=0.05,
              patch_peak=0.114 * 8 / n * p["richness"],
              patch_life=0.0 if p["_patch_forever"] else p["patch_life"],
              founders_near_patches=True, social_sensing=True, communication=True,
              seasons=p["season_amp"] >= 0.05, season_amplitude=p["season_amp"],
              season_period=int(p["season_period"]),
              hazards=p["hazard_frac"] >= 0.005, hazard_fraction=max(p["hazard_frac"], 0.005),
              cofood=False, cofood_spawn_prob=0.0, memory=False,
              share_multiplier=1.0,  # gifts break even: no energy from nothing
              buried=p["n_soil"] > 0, n_soil=max(1, p["n_soil"]), buried_base=p["buried_base"],
              gut_init_max=0.2, gut_mut=0.05, gut_q=p["gut_q"],
              predators=p["n_pred"] >= 5, n_pred=max(5, p["n_pred"]), pred_delay=1000,
              catch_p=p["catch_p"], kill_gain=p["kill_gain"], pred_evolve=True,
              pred_immigrants="bank", immigrant_every=200,
              shelter_frac=p["shelter_frac"] if p["shelter_frac"] >= 0.003 else 0.0)
    return kw


def describe(u):
    p = raw_params(u)
    bits = [f"{p['n_patches']} food clusters x{p['richness']:.2f}",
            f"drift {p['patch_speed']:.3f}",
            "clusters permanent" if p["_patch_forever"] else f"cluster life {p['patch_life']:.0f}"]
    bits.append(f"seasons {p['season_amp']:.2f}/{p['season_period']:.0f}" if p["season_amp"] >= 0.05 else "no seasons")
    bits.append(f"poison {p['hazard_frac']:.3f}" if p["hazard_frac"] >= 0.005 else "no poison")
    bits.append(f"{p['n_soil']} buried patches (base {p['buried_base']:.2f})" if p["n_soil"] > 0 else "no buried food")
    bits.append(f"{p['n_pred']} predators (catch {p['catch_p']:.2f}, gain {p['kill_gain']:.1f})"
                if p["n_pred"] >= 5 else "no predators")
    bits.append(f"shelters {p['shelter_frac']:.3f}" if p["shelter_frac"] >= 0.003 else "no shelters")
    bits.append(f"gut q {p['gut_q']:.2f}")
    return ", ".join(bits)


# ----------------------------------------------------------------------------- rendering
_PROJ = None
_REF = None
_PROJ2 = None
_REF2 = None


def _gene_vectors(sim, idx):
    w2 = sim.w2[idx].reshape(len(idx), -1)
    b2 = sim.b2[idx]
    parts = [w2, b2]
    if hasattr(sim, "gut"):
        parts.append(np.repeat(sim.gut[idx][:, None], 8, 1) * 3.0)  # gut counts as 8 genes
    return np.concatenate(parts, 1).astype(np.float64)


def _colour_setup(sim):
    """Fixed projection and scale, the same for every world (taken from the seed-1 founders)."""
    global _PROJ, _REF, _PROJ2, _REF2
    if _PROJ is not None:
        return
    idx = np.flatnonzero(sim.alive)
    g = _gene_vectors(sim, idx)
    rng = np.random.default_rng(12345)
    _PROJ = rng.normal(0, 1, (g.shape[1], 3)) / np.sqrt(g.shape[1])
    pr = g @ _PROJ
    _REF = (pr.mean(0), pr.std(0) + 1e-9)
    _PROJ2 = rng.normal(0, 1, (g.shape[1], 2)) / np.sqrt(g.shape[1])
    q = g @ _PROJ2
    _REF2 = (q.mean(0), q.std(0) + 1e-9)


def render(sim, px=2, idx=None, pidx=None):
    """Two pictures of the creatures only (the food and soil are left out, so a world
    cannot look new just because its food moves):
      space: where every creature stands (gene colour) and every predator (red), on black
      genes: every creature placed by two fixed projections of its genes (same colours);
             a new species is a new cluster, an evolving lineage a moving cloud."""
    S = sim.size
    space = np.zeros((S, S, 3))
    genes = np.zeros((S, S, 3))
    if idx is None:
        idx = np.flatnonzero(sim.alive)
    if len(idx):
        pr = (_gene_vectors(sim, idx) @ _PROJ - _REF[0]) / (2.0 * _REF[1])
        col = 0.30 + 0.70 * (0.5 + 0.5 * np.tanh(pr))
        space[sim.x[idx], sim.y[idx]] = col
        q = (_gene_vectors(sim, idx) @ _PROJ2 - _REF2[0]) / _REF2[1]   # founders: mean 0, sd 1
        gx = np.clip(np.round(S / 2 + q[:, 0] * S / 10), 0, S - 1).astype(int)  # +-5 founder sd fills the panel
        gy = np.clip(np.round(S / 2 + q[:, 1] * S / 10), 0, S - 1).astype(int)
        np.add.at(genes, (gx, gy), col * 0.6)
    if hasattr(sim, "pred") and sim.cfg.get("predators", False) and sim.n_pred:
        pi = np.arange(sim.n_pred) if pidx is None else pidx
        space[sim.pred["x"][pi], sim.pred["y"][pi]] = (1.0, 0.08, 0.08)
    out = []
    for img in (space, genes):
        img = np.repeat(np.repeat(np.clip(img, 0, 1), px, 0), px, 1)
        out.append((img * 255).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------- rollouts
_FOUNDERS = {}


def _founders(seed):
    if seed not in _FOUNDERS:
        if STAGE13 not in sys.path:
            sys.path.insert(0, os.path.abspath(STAGE13))
        from seed_from_stage5 import founders
        from engine import founders_from_old_agents
        _FOUNDERS.clear()
        _FOUNDERS[seed] = founders_from_old_agents(founders(300, seed, 48))
    return _FOUNDERS[seed]


def worker_init():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def make_sim(kw, seed):
    worker_init()
    from ways import WaysSim
    return WaysSim(seed=seed, founders=_founders(seed), **kw)


def rollout(job):
    """job: dict(kw=..., seed=..., ticks=..., every=..., pop_cap=..., min_pop=...).

    At every picture two disjoint random samples of n_draw creatures (and n_draw_pred
    predators) are drawn on their own: A and B, in both views. Same number of dots in every
    picture, so population size cannot show as change. A vs B gives the noise floor: how
    different two pictures of the very same moment look. A world below min_pop creatures
    (default 200 = 2 x n_draw) counts as collapsed and freezes, like a dead one.
    The first burn_in ticks (default 3000) are a settling period: only extinction ends the
    run there, and pictures start after it ((ticks - burn_in) / every of them).
    Returns frames (full space view, for display), A/B pictures of both views, populations."""
    import time
    t0 = time.time()
    kw, seed, ticks, every = job["kw"], job["seed"], job["ticks"], job["every"]
    pop_cap, min_pop = job.get("pop_cap", 12000), job.get("min_pop", 200)
    n_draw, n_draw_pred = job.get("n_draw", 100), job.get("n_draw_pred", 10)
    sim = make_sim(kw, seed)
    _colour_setup(_ref_sim() if _PROJ is None else sim)
    split_rng = np.random.default_rng([int(seed), 99])  # own stream: the world runs exactly as without pictures
    pics = {k: [] for k in ("full", "sA", "sB", "gA", "gB")}
    pops, preds = [], []
    status = "ok"
    # settling period: the 300 founders often dip below min_pop before a world settles, so
    # for the first burn_in ticks only extinction ends a run, and nothing is pictured
    burn_in = int(job.get("burn_in", 3000))
    while sim.tick < burn_in and sim.population > 0:
        sim.step()
    n_frames = max(2, (ticks - burn_in) // every)
    while len(pops) < n_frames:
        for _ in range(every):
            sim.step()
            if sim.population < min_pop or sim.population > pop_cap:
                break
        if sim.population == 0:
            status = f"extinct@{sim.tick}"
        elif sim.population < min_pop:
            status = f"collapsed@{sim.tick}"
        elif sim.population > pop_cap:
            status = f"exploded@{sim.tick}"
        idx = np.flatnonzero(sim.alive)
        perm = split_rng.permutation(len(idx))
        npred = sim.n_pred if hasattr(sim, "pred") and sim.cfg.get("predators", False) else 0
        pp = split_rng.permutation(npred)
        # each half shows the same number of creatures (and predators) in every picture, so a
        # growing or shrinking population cannot look like change; only where they are and what
        # their genes are can
        h = min(n_draw, len(idx) // 2)
        hA, hB = idx[np.sort(perm[:h])], idx[np.sort(perm[h:2 * h])]
        ph = min(n_draw_pred, npred // 2)
        pA, pB = pp[:ph], pp[ph:2 * ph]
        pics["full"].append(render(sim)[0])
        sA, gA = render(sim, idx=hA, pidx=pA)
        sB, gB = render(sim, idx=hB, pidx=pB)
        pics["sA"].append(sA); pics["sB"].append(sB); pics["gA"].append(gA); pics["gB"].append(gB)
        pops.append(sim.population)
        preds.append(npred)
        if status != "ok":
            break
    while len(pops) < n_frames:  # a dead, collapsed or exploded world stays frozen: nothing new
        for k in pics:
            pics[k].append(pics[k][-1])
        pops.append(pops[-1]); preds.append(preds[-1])
    out = {k: np.stack(v) for k, v in pics.items()}
    out.update(pops=pops, preds=preds, status=status, tick=sim.tick, secs=time.time() - t0,
               gens=int(sim.gen[sim.alive].max()) if sim.population else 0)
    return out


def _ref_sim():
    """Seed-1 founders in a plain world: the colour reference, identical in every process."""
    return make_sim(decode(np.full(D, 0.5)), 1)
