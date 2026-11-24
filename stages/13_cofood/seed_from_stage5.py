"""
Build stage 6's founding population from stage 5's evolved agents (the
adopted hazards-on arm), so stage 6 starts from foragers that already cope
with seasons and poison, as real lineages gain new traits on top of old ones.

Brain surgery (identical for both arms):
  - inputs: 50 zero rows inserted at index 100 for the kin and predator
    patches, so at the start those inputs have no effect at all
  - outputs: new "share", "mate" and "mark" actions, small random weights and bias
    -2.0, so they start rare and selection decides whether they grow
Each agent gets a fresh random family tag; relatedness builds up as offspring
inherit tags. Generation counters restart at 0 for this stage.
"""
import pickle
import numpy as np
from agent import Agent
from brain import Brain, HIDDEN_SIZE

import os as _os
STAGE5_CKPT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "05_hazards",
                           "runs", "hazards_on_seed1", "checkpoint_23500.pkl")
KIN_ROW_INDEX = 100  # food 0-24, crowd 25-49, signal 50-74, hazard 75-99, kin 100-124,
# predator 125-149, marker 150-174, signal2 175-199, cofood 200-224, extra 225-226


def founders(n, seed, size):
    with open(STAGE5_CKPT, "rb") as f:
        old = pickle.load(f)["agents"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(old), size=min(n, len(old)), replace=False)
    out = []
    for i in pick:
        ob = old[i].brain
        # zero rows for the kin patch AND the predator patch (stage 8): 50 new inputs
        # zero rows for the kin, predator AND marker patches: 75 new inputs
        w1 = np.insert(ob.w1, KIN_ROW_INDEX, np.zeros((75, HIDDEN_SIZE)), axis=0)
        # stage 12: 25 zero rows for the second-symbol patch, just before energy/age
        w1 = np.insert(w1, w1.shape[0] - 2, np.zeros((25, HIDDEN_SIZE)), axis=0)
        # stage 13: 25 zero rows for the shared-food patch, also just before energy/age
        w1 = np.insert(w1, w1.shape[0] - 2, np.zeros((25, HIDDEN_SIZE)), axis=0)
        w2 = np.concatenate([ob.w2, rng.normal(0, 0.1, (HIDDEN_SIZE, 3))], axis=1)
        b2 = np.append(ob.b2, [-2.0, -2.0, -2.0])
        brain = Brain(w1, ob.b1.copy(), w2, b2, ob.wh.copy(), rng=rng)  # seeded: identical founders in both arms
        out.append(Agent(int(rng.integers(size)), int(rng.integers(size)), brain=brain,
                         energy=5.0, generation=0, rng=rng, tag=rng.normal(0, 1, 4)))
    return out
