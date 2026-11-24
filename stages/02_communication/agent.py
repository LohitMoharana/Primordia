"""
An agent: a body with energy and a lifespan, driven entirely by its Brain.
We never tell it to seek food or avoid starving — that has to be discovered
by selection, because agents that don't do it die without reproducing.
"""
import numpy as np
from brain import Brain

MAX_ENERGY = 10.0
START_ENERGY = 5.0
MAX_AGE = 500            # hard lifespan cap, even if well-fed
MOVE_COST = 0.03
IDLE_COST = 0.01
SIGNAL_COST = 0.015  # a signal has to be worth more than its cost to persist
EAT_GAIN = 0.5
REPRODUCE_COST = 4.0      # energy spent by parent
REPRODUCE_MIN_ENERGY = 6.0
SENSE_RADIUS = 2


class Agent:
    _next_id = 0

    def __init__(self, x, y, brain=None, energy=START_ENERGY, generation=0, parent_id=None, rng=None):
        self.id = Agent._next_id
        Agent._next_id += 1
        self.x, self.y = x, y
        self.brain = brain or Brain(rng=rng)
        self.energy = energy
        self.age = 0
        self.generation = generation
        self.parent_id = parent_id
        self.alive = True
        self.children = 0

    def sense(self, world):
        food_patch = world.local_view(self.x, self.y, radius=SENSE_RADIUS).flatten()
        if world.social_sensing:
            agent_patch = world.local_view(self.x, self.y, radius=SENSE_RADIUS,
                                            grid=world.occupancy).flatten()
            # subtract self out of the center cell so an agent isn't counting its own presence
            center = len(agent_patch) // 2
            agent_patch = agent_patch.copy()
            agent_patch[center] = max(0.0, agent_patch[center] - 1)
        else:
            # ablation: same input SLOT exists, always zero — brain shape unchanged,
            # information removed. This is what makes the comparison fair.
            agent_patch = np.zeros_like(food_patch)
        signal_patch = world.local_view(self.x, self.y, radius=SENSE_RADIUS,
                                         grid=world.signal).flatten()
        extra = np.array([self.energy / MAX_ENERGY, self.age / MAX_AGE])
        return np.concatenate([food_patch, agent_patch, signal_patch, extra])

    def step(self, world, rng):
        """One tick: sense, act, pay the cost of living, maybe die."""
        self.age += 1
        obs = self.sense(world)
        action = self.brain.act(obs, rng)

        child = None
        if action == "stay":
            self.energy -= IDLE_COST
        elif action in ("N", "S", "E", "W"):
            dx, dy = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}[action]
            self.x, self.y = world.wrap(self.x + dx, self.y + dy)
            self.energy -= MOVE_COST
        elif action == "eat":
            gained = world.take_food(self.x, self.y, EAT_GAIN)
            self.energy += gained
            self.energy -= IDLE_COST
        elif action == "reproduce":
            self.energy -= IDLE_COST
            if self.energy >= REPRODUCE_MIN_ENERGY:
                self.energy -= REPRODUCE_COST
                child_brain = self.brain.mutate(rng)
                child = Agent(self.x, self.y, brain=child_brain,
                               energy=REPRODUCE_COST * 0.5,
                               generation=self.generation + 1,
                               parent_id=self.id, rng=rng)
                self.children += 1
        elif action.startswith("signal_"):
            self.energy -= SIGNAL_COST  # possible to send, never free

        self.energy = min(self.energy, MAX_ENERGY)
        if self.energy <= 0 or self.age >= MAX_AGE:
            self.alive = False

        self.last_action = action  # exposed so a viewer can report what happened, nothing more
        return child
