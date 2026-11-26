"""
MODES on Primordia: the published open-ended-evolution metrics, measured on our creatures.

Dolson, Vostinar, Wiser & Ofria 2019, "The MODES Toolbox: Measurements of Open-Ended
Dynamics in Evolving Systems", Artificial Life 25(1):50-73. Reference code: OEETracker in
Empirical, include/emp/Evolve/OEE.hpp (github.com/devosoft/Empirical, MIT). This file
follows that code step by step:

  Update(t)          every `resolution` ticks: snapshot the living (taxon, skeleton, count);
                     keep the last interval/resolution + 1 snapshots
  CoalescenceFilter  persistence filter: of the oldest kept snapshot (taken `interval` ago),
                     keep only taxa that are ancestors of creatures alive now
                     (GetCanopyExtantRoots); group them by skeleton, summing counts
  CalcStats          on the filtered set:
                     change     = skeletons not in the previous filtered set
                     novelty    = skeletons never seen before in the run
                     diversity  = Shannon entropy (bits) of the skeleton counts
                     complexity = highest complexity among the skeletons

What each system must define (the paper leaves it to the system):
  taxon     here: one creature (weights are continuous, so every birth is a new genotype;
            the family tree comes from the engine's birth records: id, parent)
  skeleton  the part of the genome that matters for what the creature does, without neutral
            variation. Here: a behavioural strategy. The brain is shown 32 fixed situations
            (probe_inputs.npy, recorded once from reference worlds: 16 ordinary, 12 dangerous,
            4 with a predator next to it); its most likely action is sorted into 8 kinds (stay,
            move, eat, reproduce, signal, share, mate, mark/dig); the skeleton is the kind it
            picks most in each of the 3 contexts (see skeletons() for why this resolution).
  complexity  steps of behaviour in the strategy: each different basic kind counts 1; in worlds
            with evolvable actions (actions.py) each different evolved action chain is its own
            kind (coded by its content) and counts its length.
"""
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_FILE = os.path.join(HERE, "probe_inputs.npy")

# engine ACTIONS: stay N S E W eat reproduce signal_1..4 share mate mark
ACTION_KIND = np.array([0, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7])
KIND_NAMES = ["stay", "move", "eat", "reproduce", "signal", "share", "mate", "mark/dig"]


CONTEXTS = (slice(0, 16), slice(20, 32), slice(16, 20))  # ordinary world, dangerous world, predator adjacent
CONTEXT_NAMES = ["ordinary", "dangerous", "predator next to it"]


def skeletons(sim, idx, probes):
    """MODES skeleton of creatures idx: in each of 3 contexts, the kind of action it chooses most
    often over that context's probe situations (ties: lower kind). 8^3 = 512 possible strategies.
    Chosen by heritability: in a test population a child kept its parent's skeleton in 70% of
    births (the engine's mutation changes 10% of all weights), against 5% for the full 32-situation
    fingerprint, so mutation noise is mostly left out, as MODES' skeletonisation intends."""
    raw = actions(sim, idx, probes, raw=True)
    if raw.max(initial=0) < len(ACTION_KIND):  # no evolved actions in use: the 8 kinds
        a = ACTION_KIND[raw]
        out = np.zeros((len(idx), len(CONTEXTS)), np.int64)
        for j, sl in enumerate(CONTEXTS):
            c = np.stack([(a[:, sl] == k).sum(1) for k in range(len(KIND_NAMES))], 1)
            out[:, j] = c.argmax(1)
        return out
    # evolved actions (actions.py): a creature using one of its own action chains is coded by the
    # chain's content, so different chains are different kinds of behaviour
    code = np.where(raw < len(ACTION_KIND), ACTION_KIND[np.minimum(raw, len(ACTION_KIND) - 1)], 0).astype(np.int64)
    rows, cols = np.nonzero(raw >= len(ACTION_KIND))
    for i, q in zip(rows, cols):
        k = raw[i, q] - len(ACTION_KIND)
        steps = sim.mac[idx[i], k, :sim.mac_len[idx[i], k]]
        code[i, q] = MACRO_BASE + int(sum(int(st + 1) * 15 ** p for p, st in enumerate(steps)))
    out = np.zeros((len(idx), len(CONTEXTS)), np.int64)
    for j, sl in enumerate(CONTEXTS):
        for i in range(len(idx)):
            v, c = np.unique(code[i, sl], return_counts=True)
            out[i, j] = v[np.argmax(c)]  # most chosen; ties: lowest code (as argmax above)
    return out


MACRO_BASE = 100  # skeleton codes >= this are evolved action chains (actions.py)


def actions(sim, idx, probes, raw=False):
    """Most likely action of creatures idx in each probe situation (n x 32): its kind, or with
    raw=True the output index itself (>= 14: an evolved action chain)."""
    X = probes.astype(np.float64)                                   # K x I
    h = np.tanh(np.einsum("ki,nih->nkh", X, sim.w1[idx].astype(np.float64)) + sim.b1[idx][:, None, :])
    out = np.einsum("nkh,nho->nko", h, sim.w2[idx].astype(np.float64)) + sim.b2[idx][:, None, :]
    a = out.argmax(-1)
    return a if raw else ACTION_KIND[np.minimum(a, len(ACTION_KIND) - 1)].astype(np.int8)  # n x K


def complexity(skel):
    """How many steps of behaviour the strategy uses across the 3 contexts: each different basic
    kind counts 1, each different evolved chain counts its length (so this can grow with chains)."""
    tot = 0
    for v in set(skel.tolist()):
        if v < MACRO_BASE:
            tot += 1
        else:
            x, n = v - MACRO_BASE, 0
            while x:
                x //= 15
                n += 1
            tot += n
    return tot


def entropy_bits(counts):
    tot = float(sum(counts))
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0)


class ModesTracker:
    """OEETracker for one Primordia run. Call attach(sim) once, then update(sim) after
    every step (cheap: it only reads birth records), and it snapshots every `resolution` ticks."""

    def __init__(self, probes, resolution=500, interval=5000):
        assert interval % resolution == 0, "interval must be a multiple of resolution (as in OEE.hpp)"
        self.probes, self.resolution, self.interval = probes, int(resolution), int(interval)
        self.parent = {}          # creature id -> parent id (-1 for founders)
        self.snapshots = []       # list of (time, {id: skeleton bytes})
        self.prev_coal = {}
        self.seen = set()
        self.rows = []            # (time, change, novelty, diversity, complexity, filter on)
        self.brain_rows = []      # (time, mean units, max units, mean units of persistent roots)
        self.units = {}           # creature id -> brain units (worlds with evolving brain size)
        self.macro_rows = []      # (time, macro statistics) in worlds with evolvable actions
        self.eco_rows = []        # (time, remains statistics) in worlds with remains (eco.py)

    def attach(self, sim):
        idx = np.flatnonzero(sim.alive)
        for a in sim.aid[idx]:
            self.parent[int(a)] = -1
        sim._births = []          # the engine appends every birth here (tick, id, parent, gen, lineage)

    def _drain(self, sim):
        if sim._births:
            b = np.concatenate(sim._births)
            for a, p in zip(b[:, 1].tolist(), b[:, 2].tolist()):
                self.parent[a] = p
            sim._births = []

    def update(self, sim):
        self._drain(sim)
        if sim.tick % self.resolution:
            return
        idx = np.flatnonzero(sim.alive)
        sk = skeletons(sim, idx, self.probes) if len(idx) else np.zeros((0, len(CONTEXTS)), np.int64)
        self.snapshots.append((sim.tick, {int(a): s.tobytes() for a, s in zip(sim.aid[idx], sk)}))
        if hasattr(sim, "nact"):
            # remember brain sizes only for creatures in kept snapshots (bounded memory)
            keep = set().union(*[set(d) for _, d in self.snapshots])
            self.units = {a: u for a, u in self.units.items() if a in keep}
            self.units.update({int(a): int(u) for a, u in zip(sim.aid[idx], sim.nact[idx])})
        if len(self.snapshots) > self.interval // self.resolution + 1:
            self.snapshots.pop(0)
        self._calc(sim, set(int(a) for a in sim.aid[idx]))

    def _canopy_roots(self, then_ids, alive_now):
        """Creatures alive at the old snapshot that have descendants alive now (GetCanopyExtantRoots)."""
        roots, memo = set(), {}
        for a in alive_now:
            path, x = [], a
            while x != -1 and x not in then_ids and x not in memo:
                path.append(x)
                x = self.parent.get(x, -1)
            r = memo.get(x, x if x in then_ids else -1)
            for y in path:
                memo[y] = r
            if r != -1:
                roots.add(r)
        return roots

    def _calc(self, sim, alive_now):
        coal = {}
        active = len(self.snapshots) > self.interval // self.resolution
        self._last_roots = []
        if active:
            t0, then = self.snapshots[0]
            self._last_roots = list(self._canopy_roots(then, alive_now))
            for a in self._last_roots:
                coal[then[a]] = coal.get(then[a], 0) + 1
        change = novelty = 0
        most = 0
        div = entropy_bits(list(coal.values())) if coal else 0.0
        for s in coal:
            if s not in self.prev_coal:
                change += 1
            if s not in self.seen:
                novelty += 1
                self.seen.add(s)
            most = max(most, complexity(np.frombuffer(s, np.int64)))
        self.rows.append((sim.tick, change, novelty, div, most, active))
        if hasattr(sim, "eco_summary"):
            d = sim._dig_counts - getattr(self, "_dig0", 0)
            self._dig0 = sim._dig_counts
            es = sim.eco_summary()
            es["dig_rate"] = d / max(1, self.resolution * max(1, len(alive_now)))
            self.eco_rows.append((sim.tick, es))
        if hasattr(sim, "macro_summary"):
            self.macro_rows.append((sim.tick, sim.macro_summary()))
        if hasattr(sim, "nact"):  # brain sizes: everyone alive, and the persistent lineages' roots
            idx = np.flatnonzero(sim.alive)
            na = sim.nact[idx]
            roots_units = [self.units[a] for a in getattr(self, "_last_roots", []) if a in self.units]
            self.brain_rows.append((sim.tick, float(na.mean()) if len(na) else 0.0, int(na.max()) if len(na) else 0,
                                    float(np.mean(roots_units)) if roots_units else 0.0))
        self.prev_coal = coal

    def summary(self):
        """Per-run numbers over the snapshots where the filter is active."""
        r = [x for x in self.rows if x[5]][1:]  # the first filtered snapshot counts everything as change/new
        if not r:
            return dict(change=0.0, novelty=0.0, diversity=0.0, complexity=0.0, snapshots=0,
                        units_mean=0.0, units_max=0, units_persistent=0.0)
        a = np.array([x[1:5] for x in r], float)
        out = dict(change=float(a[:, 0].mean()), novelty=float(a[:, 1].sum()),
                   diversity=float(a[:, 2].mean()), complexity=float(a[:, 3].max()), snapshots=len(r))
        if self.eco_rows:
            m = [x[1] for x in self.eco_rows]
            late = m[len(m) // 2:]
            out.update(scavengers=m[-1]["scavengers"], gut_mean=m[-1]["gut_mean"],
                       dig_rate=float(np.mean([x["dig_rate"] for x in late])),
                       remains_left=float(np.mean([x["remains_total"] for x in late])))
        if self.macro_rows:
            m = [x[1] for x in self.macro_rows]
            out.update(macros_owned=m[-1]["macros_owned"], macro_len_mean=m[-1]["macro_len_mean"],
                       macro_len_max=max(x["macro_len_max"] for x in m),
                       macro_share=float(np.mean([x["macro_share"] for x in m[len(m) // 2:]])))
        if self.brain_rows:
            b = self.brain_rows[-1]
            out.update(units_mean=b[1], units_max=b[2], units_persistent=b[3])
        return out


# ----------------------------------------------------------------------------- probe situations
def make_probes(n_per_world=16, seed=7):
    """32 fixed situations, recorded once: 16 from the healthy default world, 16 from a world
    with predators, poison, shelters and buried food (so every sense is sometimes active)."""
    import sys
    sys.path.insert(0, HERE)
    import asal_world as AW
    rng = np.random.default_rng(seed)
    harsh = AW.default_start().copy()
    for k, v in (("n_pred", 0.3), ("hazard_frac", 0.4), ("shelter_frac", 0.5), ("n_soil", 0.6), ("season_amp", 0.5)):
        harsh[AW.NAMES.index(k)] = v
    out = []
    for u in (AW.default_start(), harsh):
        sim = AW.make_sim(AW.decode(u), 1)
        for _ in range(2000):
            sim.step()
        idx = np.flatnonzero(sim.alive)
        pick = rng.choice(idx, n_per_world, replace=False)
        X = sense_of(sim, pick)
        out.append(X)
    P = np.concatenate(out).astype(np.float32)
    # predators are rarely within sight of a random creature: in 4 of the harsh-world situations,
    # put one predator in a neighbouring cell (predator slot = inputs 125-149, 5x5 patch, centre 12)
    for j, cell in zip(range(n_per_world, n_per_world + 4), (7, 11, 13, 17)):
        P[j, 125 + cell] = 1.0
    np.save(PROBE_FILE, P)
    return P


def sense_of(sim, idx):
    """The engine's own sensory input for creatures idx at this moment (as in a normal step)."""
    from engine import N_SYMBOLS  # noqa: F401  (engine import also sets up kernels)
    S = sim.size
    occ = np.zeros((S, S))
    alive = np.flatnonzero(sim.alive)
    np.add.at(occ, (sim.x[alive], sim.y[alive]), 1.0)
    sig = np.zeros((S, S))
    return sim.sense(idx, occ, sig, np.zeros((S, S)))


def load_probes():
    if not os.path.exists(PROBE_FILE):
        return make_probes()
    return np.load(PROBE_FILE)
