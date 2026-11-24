"""
An agent: a body with energy and a lifespan, driven entirely by its Brain.
We never tell it to seek food or avoid starving — that has to be discovered
by selection, because agents that don't do it die without reproducing.
"""
import numpy as np
from brain import Brain, ACTIONS

MAX_ENERGY = 10.0
START_ENERGY = 5.0
MAX_AGE = 500            # hard lifespan cap, even if well-fed
MOVE_COST = 0.03
IDLE_COST = 0.01
SIGNAL_COST = 0.015
MARK_COST = 0.015
SELFPRED_COST = 0.03  # per unit of prediction error (error is 0..1); a move costs 0.03  # same as a signal: a mark has to earn its place
HAZARD_DRAIN = 0.15
SHARE_AMOUNT = 0.5   # energy given to a neighbour by the share action
SHARE_COST = 0.02    # overhead on top of the gift, so sharing is never free
SHARE_MULTIPLIER = 2.0  # 6b: receiver gets 2x what the giver gives up (b/c = 2), so
                        # Hamilton's rule r*b > c can hold for close relatives
TAG_MUTATION = 0.05  # how much a child's family tag differs from its parent's  # per tick spent on a poison cell: 5x the cost of a move  # a signal has to be worth more than its cost to persist
EAT_GAIN = 0.5
REPRODUCE_COST = 4.0      # energy spent by parent
REPRODUCE_MIN_ENERGY = 6.0
SENSE_RADIUS = 2


class Agent:
    _next_id = 0

    def __init__(self, x, y, brain=None, energy=START_ENERGY, generation=0, parent_id=None, rng=None, tag=None):
        self.id = Agent._next_id
        Agent._next_id += 1
        self.x, self.y = x, y
        self.brain = brain or Brain(rng=rng)
        self.energy = energy
        self.age = 0
        self.generation = generation
        self.parent_id = parent_id
        self.parent2_id = None  # set for two-parent offspring
        self.alive = True
        self.children = 0
        self.h = None
        self.pred = None      # stage 11: last tick's prediction of this tick's energy
        self.err_sum = 0.0    # lifetime sum of |prediction error|
        self.err_n = 0
        self.w2_life = None  # stage 10: output weights as adjusted during this life
        rng_ = rng or np.random.default_rng()
        # heritable family tag: close relatives have close tags
        self.tag = tag if tag is not None else rng_.normal(0, 1, 4)  # working memory carried between ticks (starts empty at birth)

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
        hazard_patch = world.local_view(self.x, self.y, radius=SENSE_RADIUS,
                                         grid=world.hazard).flatten()
        kin_patch = world.kin_patch(self.x, self.y, self.tag, radius=SENSE_RADIUS)
        predator_patch = world.local_view(self.x, self.y, radius=SENSE_RADIUS,
                                           grid=world.predator).flatten()
        if world.markers_on:
            marker_patch = np.minimum(world.local_view(self.x, self.y, radius=SENSE_RADIUS,
                                                       grid=world.marker).flatten(), 5.0) / 5.0
        else:
            marker_patch = np.zeros(25)
        extra = np.array([self.energy / MAX_ENERGY, self.age / MAX_AGE])
        return np.concatenate([food_patch, agent_patch, signal_patch, hazard_patch, kin_patch,
                               predator_patch, marker_patch, extra])

    def step(self, world, rng):
        """One tick: sense, act, pay the cost of living, maybe die."""
        self.age += 1
        if self.pred is not None:
            err = abs(self.pred - self.energy / MAX_ENERGY)
            self.err_sum += err
            self.err_n += 1
            if world.selfpred:
                self.energy -= SELFPRED_COST * err  # being wrong about yourself costs
        obs = self.sense(world)
        # memory-off arm: wipe the carried state every tick (same brain, no memory)
        h_prev = self.h if world.memory else None
        e_before = self.energy
        if world.learning and self.w2_life is None:
            self.w2_life = self.brain.w2.copy()  # start life from the inherited weights
        action, self.h = self.brain.act(obs, rng, h_prev,
                                        w2=self.w2_life if world.learning else None)
        self.pred = self.brain.predict(self.h)  # "what will my energy be next tick?"

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
                               parent_id=self.id, rng=rng,
                               tag=self.tag + rng.normal(0, TAG_MUTATION, 4))
                self.children += 1
        elif action == "mark":
            self.energy -= MARK_COST
            world.marker[self.x, self.y] += 1.0
        elif action == "mate":
            # Two-parent reproduction. Same cost as asexual reproduce, paid by
            # the initiator alone; the partner pays nothing. So the ONLY
            # difference from "reproduce" is that the child's genome mixes
            # two parents. In the OFF arm the action just fails.
            self.energy -= IDLE_COST
            if world.sexual and self.energy >= REPRODUCE_MIN_ENERGY:
                mates = [b for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                         for b in world.agents_by_cell.get(world.wrap(self.x + dx, self.y + dy), [])
                         if b is not self and b.alive]
                if mates:
                    m = mates[rng.integers(len(mates))]
                    self.energy -= REPRODUCE_COST
                    child_brain = self.brain.crossover(m.brain, rng).mutate(rng)
                    child = Agent(self.x, self.y, brain=child_brain,
                                   energy=REPRODUCE_COST * 0.5,
                                   generation=max(self.generation, m.generation) + 1,
                                   parent_id=self.id, rng=rng,
                                   tag=(self.tag + m.tag) / 2 + rng.normal(0, TAG_MUTATION, 4))
                    child.parent2_id = m.id
                    self.children += 1
        elif action.startswith("signal_"):
            self.energy -= SIGNAL_COST  # possible to send, never free
        elif action == "share":
            # gift energy to a RANDOM neighbour within radius 1 (same rule in both
            # arms). Who benefits depends only on who is nearby; kin perception
            # can only matter through WHEN an agent chooses to share.
            self.energy -= SHARE_COST
            nbrs = [b for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for b in world.agents_by_cell.get(world.wrap(self.x + dx, self.y + dy), [])
                    if b is not self and b.alive]
            if nbrs and self.energy > SHARE_AMOUNT:
                b = nbrs[rng.integers(len(nbrs))]
                self.energy -= SHARE_AMOUNT
                b.energy = min(b.energy + SHARE_AMOUNT * SHARE_MULTIPLIER, MAX_ENERGY)
                self.last_share_similarity = float(np.exp(-np.linalg.norm(b.tag - self.tag)))

        if world.hazard[self.x, self.y]:
            self.energy -= HAZARD_DRAIN  # poisoned: wherever it ended up this tick
        self.energy = min(self.energy, MAX_ENERGY)
        if world.learning and self.w2_life is not None:
            # Reward-modulated learning: if energy rose this tick, make the action
            # just taken more likely in similar situations; if it fell, less.
            # Only this agent's lifetime copy changes; children get the genome.
            reward = self.energy - e_before
            col = ACTIONS.index(action)
            self.w2_life[:, col] += self.brain.lr * reward * self.h
            np.clip(self.w2_life, -5.0, 5.0, out=self.w2_life)
        if self.energy <= 0 or self.age >= MAX_AGE:
            self.alive = False

        self.last_action = action  # exposed so a viewer can report what happened, nothing more
        return child
