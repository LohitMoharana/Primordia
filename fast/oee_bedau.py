"""
Bedau's evolutionary activity statistics with a neutral shadow, on Primordia.

Bedau & Packard 1991; Bedau, Snyder & Packard 1998 ("A classification of long-term evolutionary
dynamics"); Channon 2001/2006 (Geb, the first artificial system to pass this test). The test asks
whether new adaptive components keep appearing without bound, beyond what neutral drift alone would
produce.

Components: the same behavioural strategies MODES uses (oee_modes.skeletons). Every `resolution`
ticks the living creatures are counted by component, and each present component's activity grows by
its count (activity = how widely and how long it has lasted).

Neutral shadow (the null model, matched to the real run): the same number of creatures at every
snapshot; each shadow creature inherits its component from a random shadow creature of the previous
snapshot (so persistence is pure chance), and becomes a brand-new component with probability mu_t,
the fraction of real creatures whose strategy differs from that of their own ancestor at the previous
snapshot (so new components arise at the real rate). Real components that reach far more activity
than shadow components ever do are adaptive: selection, not chance, kept them.

The tracker only records counts; bedau_analyze.py computes the threshold (from the shadow), new
evolutionary activity over time, cumulative and mean activity, and Bedau's classification.
"""
import json

import numpy as np

import oee_modes as M


class BedauTracker:
    def __init__(self, probes, leader, resolution=500, seed=0):
        """leader: the run's main ModesTracker (shares its family tree; update it first)."""
        self.probes, self.leader, self.resolution = probes, leader, int(resolution)
        self.rng = np.random.default_rng([int(seed), 8484])
        self.prev = None          # {creature id: component} at the previous snapshot
        self.shadow = None        # array of shadow component ids at the previous snapshot
        self.next_shadow = 0
        self.names = {}           # real component (strategy bytes) -> integer id
        self.real_rows = []       # per snapshot: (component ids, counts) as compact arrays
        self.shadow_rows = []     # per snapshot: (shadow component ids, counts)
        self.ticks, self.mu = [], []

    def _ancestor_comp(self, ids_now, prev):
        """For each living creature, the component of its ancestor alive at the previous snapshot."""
        par = self.leader.parent  # read every time: the leader may replace it when pruning
        memo, out = {}, []
        for a in ids_now:
            path, x = [], a
            while x != -1 and x not in prev and x not in memo:
                path.append(x)
                x = par.get(x, -1)
            r = memo.get(x, prev.get(x)) if x != -1 else None
            for y in path:
                memo[y] = r
            out.append(r)
        return out

    def update(self, sim):
        if sim.tick % self.resolution:
            return
        idx = np.flatnonzero(sim.alive)
        if not len(idx):
            return
        sk = M.skeletons(sim, idx, self.probes)
        ids = [int(a) for a in sim.aid[idx]]
        comps = [s.tobytes() for s in sk]
        cur = dict(zip(ids, comps))
        n = len(ids)
        if self.prev is None:
            mu = 0.0
            uniq = {c: i for i, c in enumerate(dict.fromkeys(comps))}
            self.shadow = np.array([uniq[c] for c in comps])
            self.next_shadow = len(uniq)
        else:
            anc = self._ancestor_comp(ids, self.prev)
            known = [(c, a) for c, a in zip(comps, anc) if a is not None]
            mu = float(np.mean([c != a for c, a in known])) if known else 0.0
            # neutral shadow: random parents from the previous shadow, new components at rate mu
            sh = self.shadow[self.rng.integers(0, len(self.shadow), n)]
            new = self.rng.random(n) < mu
            k = int(new.sum())
            sh[new] = np.arange(self.next_shadow, self.next_shadow + k)
            self.next_shadow += k
            self.shadow = sh
        self.prev = cur
        rid = np.array([self.names.setdefault(c, len(self.names)) for c in comps], np.int64)
        v, c = np.unique(rid, return_counts=True)
        self.real_rows.append((v.astype(np.int32), c.astype(np.int32)))
        sv, sc = np.unique(self.shadow, return_counts=True)
        self.shadow_rows.append((sv.astype(np.int64), sc.astype(np.int32)))
        self.ticks.append(int(sim.tick))
        self.mu.append(mu)

    def save(self, path, **meta):
        """Compressed .npz: ticks, mu, and for real and shadow the concatenated (id, count) arrays
        with per-snapshot offsets."""
        def pack(rows):
            off = np.cumsum([0] + [len(r[0]) for r in rows])
            ids = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int64)
            cnt = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.int32)
            return ids, cnt, off
        ri, rc, ro = pack(self.real_rows)
        si, sc, so = pack(self.shadow_rows)
        np.savez_compressed(path, ticks=np.array(self.ticks), mu=np.array(self.mu), resolution=self.resolution,
                            real_ids=ri, real_counts=rc, real_off=ro, shadow_ids=si, shadow_counts=sc,
                            shadow_off=so, meta=np.array(json.dumps(meta)))
