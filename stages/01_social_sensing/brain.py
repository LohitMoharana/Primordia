"""
An agent's brain: a tiny neural network, nothing hand-coded about behavior.
Senses in, action out. All behavior comes from these weights, which are
only ever set by inheritance + mutation, never by us writing rules.
"""
import numpy as np

# Inputs: local food patch (radius=2 -> 25 cells)
#       + local agent-occupancy patch (radius=2 -> 25 cells) — how crowded nearby cells are
#       + own energy (1) + own age fraction (1)
# Nothing here tells the brain what to DO with the agent-occupancy info — whether
# crowding means competition (avoid) or safety (cluster) is left for evolution to find.
INPUT_SIZE = 25 + 25 + 2
HIDDEN_SIZE = 16
# Actions: stay, N, S, E, W, EAT, REPRODUCE  (7 actions)
OUTPUT_SIZE = 7

ACTIONS = ["stay", "N", "S", "E", "W", "eat", "reproduce"]


class Brain:
    """A simple 2-layer MLP. Weights ARE the genome."""

    def __init__(self, w1=None, b1=None, w2=None, b2=None, rng=None):
        rng = rng or np.random.default_rng()
        scale1 = np.sqrt(2.0 / INPUT_SIZE)
        scale2 = np.sqrt(2.0 / HIDDEN_SIZE)
        self.w1 = w1 if w1 is not None else rng.normal(0, scale1, (INPUT_SIZE, HIDDEN_SIZE))
        self.b1 = b1 if b1 is not None else np.zeros(HIDDEN_SIZE)
        self.w2 = w2 if w2 is not None else rng.normal(0, scale2, (HIDDEN_SIZE, OUTPUT_SIZE))
        self.b2 = b2 if b2 is not None else np.zeros(OUTPUT_SIZE)

    def act(self, sensory_input, rng):
        h = np.tanh(sensory_input @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        # softmax -> sample (not argmax), so behavior isn't rigidly deterministic
        probs = np.exp(out - out.max())
        probs /= probs.sum()
        action_idx = rng.choice(OUTPUT_SIZE, p=probs)
        return ACTIONS[action_idx]

    def mutate(self, rng, rate=0.1, strength=0.15):
        """Returns a MUTATED COPY. Parent is untouched."""
        def mutate_arr(arr):
            mask = rng.random(arr.shape) < rate
            noise = rng.normal(0, strength, arr.shape)
            return arr + mask * noise

        return Brain(
            mutate_arr(self.w1), mutate_arr(self.b1),
            mutate_arr(self.w2), mutate_arr(self.b2),
        )

    def complexity(self):
        """A crude proxy for 'how much the weights are doing' — mean abs weight."""
        return float(np.mean(np.abs(self.w1)) + np.mean(np.abs(self.w2)))
