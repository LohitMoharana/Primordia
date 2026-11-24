"""
The world: a toroidal grid with decaying, regrowing food.
No agent logic lives here — this file only knows about terrain and food.
"""
import numpy as np


class World:
    def __init__(self, size=48, food_regrow_prob=0.02, food_decay_prob=0.01,
                 max_food_per_cell=1.0, seed=0, social_sensing=True, communication=True,
                 seasons=False, season_period=600, season_amplitude=0.9, memory=False,
                 hazards=False, hazard_fraction=0.03, kin=False,
                 sexual=False, predators=False, predator_density=0.005,
                 predator_digest=15, markers=False, marker_half_life=200, learning=False,
                 selfpred=False):
        self.size = size
        # STAGE 3 SWITCH: with seasons on, the regrowth rate swings between
        # lean and plentiful over season_period ticks (about one lifetime).
        # The AVERAGE rate is the same in both arms, so "off" is not simply
        # a richer world; only the variability differs. Agents are never told
        # what season it is.
        self.seasons = seasons
        # STAGE 4 SWITCH: whether agents' working memory carries over between
        # ticks. Brain architecture is identical either way (see brain.py).
        self.memory = memory
        # STAGE 5 SWITCH: poison cells. Standing on one drains energy (see
        # agent.py). They cover ~3% of cells and move to new random spots every
        # season period, so agents can't just memorise a map. Agents in BOTH
        # arms have a hazard-sensing patch; with hazards off it always reads 0.
        self.hazards = hazards
        # STAGE 6 SWITCH: whether agents can perceive how related their
        # neighbours are (via heritable tags). Brain shape is identical either
        # way; with kin off the kin-sensing patch reads zero.
        self.kin = kin
        # STAGE 7 SWITCH: whether the "mate" action works (two-parent offspring).
        self.sexual = sexual
        # STAGE 8 SWITCH: scripted predators (they do not evolve). ~1 per 200
        # cells; each tick a predator moves with probability 0.5 (so agents are
        # faster), toward the nearest agent within radius 3 if any, else at
        # random, and eats one agent in its cell, then digests for
        # predator_digest ticks before it can eat again. (First version: 1 per
        # 100 cells, no digesting, drove the world extinct by tick 337.) Agents in BOTH arms have a
        # predator-sensing patch; with predators off it reads zero.
        # STAGE 9 SWITCH: lasting marks (external memory). The "mark" action adds
        # 1.0 to the agent's cell; marks fade, halving every marker_half_life
        # ticks. Agents in BOTH arms have a marker patch; with markers off it
        # reads zero (marking is still allowed and still costs energy).
        self.markers_on = markers
        # STAGE 10 SWITCH: lifetime learning (see agent.py); off = learning rate unused
        self.learning = learning
        # STAGE 11 SWITCH: self-prediction error costs energy (on) or nothing (off)
        self.selfpred = selfpred
        self.marker = np.zeros((size, size))
        self.marker_decay = 0.5 ** (1.0 / marker_half_life)
        self.predators_on = predators
        self.predator_density = predator_density
        self.predator_digest = predator_digest
        self.pred_digest = []
        self.pred_pos = []  # placed on the first step_predators call (needs self.rng)
        self.predator = np.zeros((size, size))
        self.kills = 0
        r = 3
        offs = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if (dx, dy) != (0, 0)]
        self._pred_offsets = sorted(offs, key=lambda o: abs(o[0]) + abs(o[1]))
        # Grace period: hazards and season swings ramp from 0 to full over the
        # first ramp_ticks, identically in both arms, so random founding brains
        # get time for selection to start (without it this stage went extinct).
        self.ramp_ticks = 0  # off: stage 6 is seeded from stage 5's evolved population
        self.tag_sum = np.zeros((size, size, 4))
        self.agents_by_cell = {}
        self.hazard_fraction = hazard_fraction
        self.hazard = np.zeros((size, size))
        self.season_period = season_period
        self.season_amplitude = season_amplitude
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.food_regrow_prob = food_regrow_prob
        self.food_decay_prob = food_decay_prob
        self.max_food = max_food_per_cell
        # ABLATION SWITCHES: brain architecture is IDENTICAL regardless of these.
        # When False, the corresponding input slot is always zero — isolating
        # whether the INFORMATION helps, not whether a bigger brain does.
        self.social_sensing = social_sensing
        self.communication = communication  # if False, signal actions are allowed
        # but senders get nothing for it except the cost, and others never see it
        # food[x, y] in [0, max_food]
        self.food = self.rng.uniform(0, max_food_per_cell, size=(size, size))
        self.food *= (self.rng.random((size, size)) < 0.3)  # patchy start
        # occupancy[x, y] = how many agents are at (x, y) right now.
        # Rebuilt fresh every tick by the simulation, before agents sense/act.
        # This is the ONLY way agents perceive each other — no hidden channel.
        self.occupancy = np.zeros((size, size))
        # signal[x, y] = symbol id (0 = none) broadcast in that cell LAST tick.
        # Lasts exactly one tick, like a call rather than a scent. We assign
        # no meaning to any symbol id ourselves.
        self.signal = np.zeros((size, size))

    def set_occupancy(self, agents):
        self.occupancy.fill(0)
        self.tag_sum.fill(0)
        self.agents_by_cell = {}
        for a in agents:
            self.occupancy[a.x, a.y] += 1
            self.tag_sum[a.x, a.y] += a.tag
            self.agents_by_cell.setdefault((a.x, a.y), []).append(a)

    def kin_patch(self, x, y, own_tag, radius=2):
        """5x5 patch: similarity exp(-distance) between own tag and the mean tag
        of agents in each nearby cell (0 where empty). Excludes the agent itself."""
        if not self.kin:
            return np.zeros((2 * radius + 1) ** 2)
        idx = np.arange(-radius, radius + 1)
        xs, ys = (x + idx) % self.size, (y + idx) % self.size
        occ = self.occupancy[np.ix_(xs, ys)].copy()
        tags = self.tag_sum[np.ix_(xs, ys)].copy()
        occ[radius, radius] -= 1
        tags[radius, radius] -= own_tag
        present = occ > 0
        mean = np.where(present[..., None], tags / np.maximum(occ, 1)[..., None], 0.0)
        sim = np.exp(-np.linalg.norm(mean - own_tag, axis=-1))
        return np.where(present, sim, 0.0).flatten()

    def step_predators(self):
        """Move predators and let them eat. Call after set_occupancy, before agents act."""
        self.predator.fill(0)
        if not self.predators_on:
            return
        if not self.pred_pos:
            n = max(1, int(self.size * self.size * self.predator_density))
            self.pred_pos = [tuple(int(v) for v in p) for p in self.rng.integers(0, self.size, (n, 2))]
            self.pred_digest = [0] * n
        new_pos = []
        for i, (px, py) in enumerate(self.pred_pos):
            if self.rng.random() < 0.5:
                target = None
                for dx, dy in self._pred_offsets:
                    if self.occupancy[(px + dx) % self.size, (py + dy) % self.size] > 0:
                        target = (dx, dy)
                        break
                if target is None:
                    target = [(1, 0), (-1, 0), (0, 1), (0, -1)][self.rng.integers(4)]
                dx, dy = target
                if abs(dx) >= abs(dy) and dx != 0:
                    px = (px + int(np.sign(dx))) % self.size
                elif dy != 0:
                    py = (py + int(np.sign(dy))) % self.size
            # ramp-in: no hunting before tick 1500 (lets the founders' boom/bust
            # settle), then the chance a hungry predator hunts rises to 1 by 3000
            hunt_p = min(1.0, max(0.0, (self.t - 1500) / 1500))
            if self.pred_digest[i] > 0:
                self.pred_digest[i] -= 1
            elif self.rng.random() < hunt_p:
                prey = [a for a in self.agents_by_cell.get((px, py), []) if a.alive]
                if prey:
                    victim = prey[self.rng.integers(len(prey))]
                    victim.alive = False
                    self.kills += 1
                    self.pred_digest[i] = self.predator_digest
            new_pos.append((px, py))
            self.predator[px, py] += 1
        self.pred_pos = new_pos

    def set_signals(self, agents):
        """Call once per tick, BEFORE agents act, using position/last_action
        from the tick before — same one-tick-delayed convention as occupancy."""
        self.signal.fill(0)
        if not self.communication:
            return  # ablation off: signals never populate, agents still pay to try
        for a in agents:
            action = getattr(a, "last_action", None)
            if action and action.startswith("signal_"):
                self.signal[a.x, a.y] = int(action.split("_")[1])

    def step(self):
        """Food decays a little each step, and empty cells occasionally regrow."""
        decay_mask = self.rng.random((self.size, self.size)) < self.food_decay_prob
        self.food[decay_mask] *= 0.5

        self.t += 1
        self.marker *= self.marker_decay
        if self.hazards and (self.t == 1 or self.t % self.season_period == 0):
            frac = self.hazard_fraction * self.harshness()
            self.hazard = (self.rng.random((self.size, self.size)) < frac).astype(float)
        p = self.current_regrow_prob()
        regrow_mask = (self.food < 0.05) & (self.rng.random((self.size, self.size)) < p)
        self.food[regrow_mask] = self.max_food

        np.clip(self.food, 0, self.max_food, out=self.food)

    def current_regrow_prob(self):
        if not self.seasons:
            return self.food_regrow_prob
        phase = 2 * np.pi * self.t / self.season_period
        return self.food_regrow_prob * (1 + self.season_amplitude * self.harshness() * np.sin(phase))

    def harshness(self):
        if not self.ramp_ticks:
            return 1.0
        return min(1.0, self.t / self.ramp_ticks)

    def wrap(self, x, y):
        return x % self.size, y % self.size

    def take_food(self, x, y, amount=0.5):
        """Agent eats at (x, y). Returns actual amount gained."""
        available = self.food[x, y]
        gained = min(available, amount)
        self.food[x, y] -= gained
        return gained

    def local_view(self, x, y, radius=1, grid=None):
        """Returns a (2r+1, 2r+1) patch centered on (x, y), toroidally wrapped.
        Defaults to the food grid; pass grid=self.occupancy to see other agents."""
        if grid is None:
            grid = self.food
        idx = np.arange(-radius, radius + 1)
        xs = (x + idx) % self.size
        ys = (y + idx) % self.size
        return grid[np.ix_(xs, ys)]
