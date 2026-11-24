"""
The world: a toroidal grid with decaying, regrowing food.
No agent logic lives here — this file only knows about terrain and food.
"""
import numpy as np


class World:
    def __init__(self, size=48, food_regrow_prob=0.02, food_decay_prob=0.01,
                 max_food_per_cell=1.0, seed=0, social_sensing=True):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.food_regrow_prob = food_regrow_prob
        self.food_decay_prob = food_decay_prob
        self.max_food = max_food_per_cell
        # ABLATION SWITCH: the brain's input size and architecture are IDENTICAL
        # either way. When False, agents still have the occupancy-sensing
        # input slot, it's just always zero — so any behavioral difference
        # between social_sensing=True/False runs is attributable to the
        # information itself, not to having a bigger or different brain.
        self.social_sensing = social_sensing
        # food[x, y] in [0, max_food]
        self.food = self.rng.uniform(0, max_food_per_cell, size=(size, size))
        self.food *= (self.rng.random((size, size)) < 0.3)  # patchy start
        # occupancy[x, y] = how many agents are at (x, y) right now.
        # Rebuilt fresh every tick by the simulation, before agents sense/act.
        # This is the ONLY way agents perceive each other — no hidden channel.
        self.occupancy = np.zeros((size, size))

    def set_occupancy(self, agents):
        self.occupancy.fill(0)
        for a in agents:
            self.occupancy[a.x, a.y] += 1

    def step(self):
        """Food decays a little each step, and empty cells occasionally regrow."""
        decay_mask = self.rng.random((self.size, self.size)) < self.food_decay_prob
        self.food[decay_mask] *= 0.5

        regrow_mask = (self.food < 0.05) & (self.rng.random((self.size, self.size)) < self.food_regrow_prob)
        self.food[regrow_mask] = self.max_food

        np.clip(self.food, 0, self.max_food, out=self.food)

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
