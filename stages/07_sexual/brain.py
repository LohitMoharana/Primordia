"""
An agent's brain: a tiny neural network, nothing hand-coded about behavior.
Senses in, action out. All behavior comes from these weights, which are
only ever set by inheritance + mutation, never by us writing rules.
"""
import numpy as np

# Inputs: food patch (25) + occupancy patch (25) + signal patch (25) + energy + age
# Nothing here tells the brain what a symbol MEANS. If "symbol 2" comes to
# correlate with "food here," that's for selection to discover, not us to wire.
N_SYMBOLS = 4
# + hazard patch (25): poison cells nearby (stage 5)
# + kin patch (25): how related nearby agents are (stage 6)
INPUT_SIZE = 25 + 25 + 25 + 25 + 25 + 2
HIDDEN_SIZE = 24
OUTPUT_SIZE = 7 + N_SYMBOLS + 2  # + share, mate

ACTIONS = ["stay", "N", "S", "E", "W", "eat", "reproduce"] + [f"signal_{i}" for i in range(1, N_SYMBOLS + 1)] + ["share", "mate"]


class Brain:
    """A small recurrent network. Weights ARE the genome.

    STAGE 4: the hidden layer feeds back into itself through wh, so what the
    agent sensed a few ticks ago can still shape what it does now — a short
    working memory. Every brain has wh in both arms; in the memory-off arm
    the carried state is wiped to zero each tick, so wh multiplies zeros and
    the brain behaves as memoryless while keeping the same architecture.
    """

    def __init__(self, w1=None, b1=None, w2=None, b2=None, wh=None, rng=None):
        rng = rng or np.random.default_rng()
        scale1 = np.sqrt(2.0 / INPUT_SIZE)
        scale2 = np.sqrt(2.0 / HIDDEN_SIZE)
        self.w1 = w1 if w1 is not None else rng.normal(0, scale1, (INPUT_SIZE, HIDDEN_SIZE))
        self.b1 = b1 if b1 is not None else np.zeros(HIDDEN_SIZE)
        self.w2 = w2 if w2 is not None else rng.normal(0, scale2, (HIDDEN_SIZE, OUTPUT_SIZE))
        self.b2 = b2 if b2 is not None else np.zeros(OUTPUT_SIZE)
        # small init so early brains are nearly memoryless; selection can grow it
        self.wh = wh if wh is not None else rng.normal(0, 0.1 / np.sqrt(HIDDEN_SIZE),
                                                       (HIDDEN_SIZE, HIDDEN_SIZE))

    def act(self, sensory_input, rng, h_prev=None):
        """Returns (action, new hidden state)."""
        if h_prev is None:
            h_prev = np.zeros(HIDDEN_SIZE)
        h = np.tanh(sensory_input @ self.w1 + h_prev @ self.wh + self.b1)
        out = h @ self.w2 + self.b2
        # softmax -> sample (not argmax), so behavior isn't rigidly deterministic
        probs = np.exp(out - out.max())
        probs /= probs.sum()
        action_idx = rng.choice(OUTPUT_SIZE, p=probs)
        return ACTIONS[action_idx], h

    def mutate(self, rng, rate=0.1, strength=0.15):
        """Returns a MUTATED COPY. Parent is untouched."""
        def mutate_arr(arr):
            mask = rng.random(arr.shape) < rate
            noise = rng.normal(0, strength, arr.shape)
            return arr + mask * noise

        return Brain(
            mutate_arr(self.w1), mutate_arr(self.b1),
            mutate_arr(self.w2), mutate_arr(self.b2),
            mutate_arr(self.wh),
        )

    def crossover(self, other, rng):
        """Uniform crossover: each weight comes from self or other at random."""
        def mix(a, c):
            return np.where(rng.random(a.shape) < 0.5, a, c)
        return Brain(mix(self.w1, other.w1), mix(self.b1, other.b1),
                     mix(self.w2, other.w2), mix(self.b2, other.b2),
                     mix(self.wh, other.wh))

    def memory_strength(self):
        """Mean |recurrent weight| — how much the brain leans on its past state."""
        return float(np.mean(np.abs(self.wh)))

    def complexity(self):
        """A crude proxy for 'how much the weights are doing' — mean abs weight."""
        return float(np.mean(np.abs(self.w1)) + np.mean(np.abs(self.w2)))
