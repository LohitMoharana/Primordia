"""
Remains: a resource that exists only because of other creatures (Soros & Stanley's condition 2).

Soros & Stanley (2014), condition 2: "evolution of individuals should create novel opportunities
to meet the minimal criterion". In nature most niches are made by other living things: a carcass
feeds scavengers, dung feeds beetles, a dead tree feeds fungi. In Primordia so far every food
source was put there by the world. Here the second food is made by the creatures themselves.

  remains   when a prey creature dies (hunger, old age or a predator), its body becomes food on the
            cell where it died: body_value + body_energy x its remaining energy. Remains rot (each
            tick they lose a fraction `rot`). They are eaten with the dig action and digested with
            the gut trade-off of the several-ways-to-live world (ways.py): a gut good at remains is
            bad at surface food, so scavenging is a way of life of its own. Remains are sensed in
            inputs 150-174 (as buried food). There is no soil: nothing else makes this food.

So the scavengers' food supply is made by everyone else's lives and deaths: more where crowds
live, where predators hunt, where poison kills; and it changes as they evolve.

  remains_source="bodies"  (ON)  remains appear where creatures die.
  remains_source="random"  (OFF, matched control) exactly the same amount appears every tick,
                           but spread over random cells: the same food, but no longer an
                           opportunity made by others' behaviour (not where they live or die).

Own random stream rng_eco (saved in checkpoints).
"""
import numpy as np

from ways import WaysSim


class EcoSim(WaysSim):
    def __init__(self, *args, remains_source="bodies", body_value=1.0, body_energy=0.5, rot=0.002,
                 remains_max=6.0, n_random_cells=8, **kw):
        kw["buried"] = True
        super().__init__(*args, **kw)
        seed = int(kw.get("seed", args[0] if args else 1))
        self.rng_eco = np.random.default_rng([seed, 6262])
        self.cfg.update(eco=True, remains_source=str(remains_source), body_value=float(body_value),
                        body_energy=float(body_energy), rot=float(rot), remains_max=float(remains_max),
                        n_random_cells=int(n_random_cells))
        self.buried[:] = 0.0          # no soil: remains only come from deaths
        self.soil_map[:] = 0.0
        self._eco_stats = dict(deaths=0, deposited=0.0)

    def step(self):
        idx = np.flatnonzero(self.alive)
        bx, by, be = self.x[idx].copy(), self.y[idx].copy(), self.energy[idx].copy()
        pos = {a: k for k, a in enumerate(self.aid[idx].tolist())}
        super().step()
        now = set(self.aid[np.flatnonzero(self.alive)].tolist())
        gone = [k for a, k in pos.items() if a not in now]
        if gone:
            g = np.array(gone)
            amount = self.cfg["body_value"] + self.cfg["body_energy"] * np.maximum(be[g], 0.0)
            self._deposit(bx[g], by[g], amount)

    def _deposit(self, x, y, amount):
        c, S = self.cfg, self.size
        total = float(amount.sum())
        self._eco_stats["deaths"] += len(amount)
        self._eco_stats["deposited"] += total
        if c["remains_source"] == "bodies":
            np.add.at(self.buried, (x, y), amount)
        else:  # matched control: the same total, on a few random cells (body-sized lumps)
            k = max(1, min(len(amount), c["n_random_cells"]))
            rx = self.rng_eco.integers(0, S, k)
            ry = self.rng_eco.integers(0, S, k)
            np.add.at(self.buried, (rx, ry), np.full(k, total / k))
        np.minimum(self.buried, c["remains_max"], out=self.buried)

    def _world_step(self, foods=None):
        super()._world_step(foods)   # soil regrowth finds no soil (soil_map is zero)
        self.buried *= 1.0 - self.cfg["rot"]

    def eco_summary(self):
        idx = np.flatnonzero(self.alive)
        g = self.gut[idx]
        st = self._eco_stats
        out = dict(scavengers=float((g > 0.5).mean()) if len(idx) else 0.0,
                   gut_mean=float(g.mean()) if len(idx) else 0.0,
                   remains_total=float(self.buried.sum()), deaths=st["deaths"], deposited=st["deposited"])
        self._eco_stats = dict(deaths=0, deposited=0.0)  # counts since the last summary
        return out

    def _checkpoint_extra(self):
        d = super()._checkpoint_extra()
        d.update(rng_eco_state=self.rng_eco.bit_generator.state, eco_stats=self._eco_stats)
        return d

    @classmethod
    def from_checkpoint(cls, path, out_dir=None):
        import pickle
        sim = super().from_checkpoint(path, out_dir=out_dir)
        with open(path, "rb") as f:
            d = pickle.load(f)
        sim.rng_eco = np.random.default_rng()
        sim.rng_eco.bit_generator.state = d["rng_eco_state"]
        sim._eco_stats = d.get("eco_stats", dict(deaths=0, deposited=0.0))
        return sim
