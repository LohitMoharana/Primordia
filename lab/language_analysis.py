"""
Stage 12 measure: does the second symbol carry information beyond the first?

    python3 language_analysis.py <stage_dir> <checkpoint_path>

For every agent in the checkpoint, in its actual situation, compute (from its
brain, no sampling noise): how likely it is to signal, its distribution over
first symbols, and its distribution over second symbols. Situations:
  food direction: which part of the 5x5 food patch holds most food
                  (here / up / down / left / right / none)
  energy: below or above half
Reports, weighted by each agent's probability of signalling (bits):
  I(situation; sym1), I(situation; sym1, sym2), and the extra information the
  second symbol adds, I(situation; sym2 | sym1) = I(s; sym1,sym2) - I(s; sym1).
Each is also given minus a shuffled-situation baseline (finite-sample bias).
Compare ON with OFF: in OFF the second symbol is never delivered, so any
structure there is drift, the neutral baseline.
"""
import os
import pickle
import sys

import numpy as np


def mi(joint):
    joint = joint / joint.sum()
    ps = joint.sum(axis=1, keepdims=True)
    pm = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (ps @ pm)[nz])).sum())


def food_dir(food):
    f = food.reshape(5, 5)
    regions = {
        0: f[2, 2],
        1: f[:2, :].sum(), 2: f[3:, :].sum(),
        3: f[:, :2].sum(), 4: f[:, 3:].sum(),
    }
    if max(regions.values()) <= 0:
        return 5
    return max(regions, key=regions.get)


def main(stage_dir, ckpt):
    sys.path.insert(0, os.path.abspath(stage_dir))
    from brain import ACTIONS, N_SYMBOLS
    with open(ckpt, "rb") as f:
        data = pickle.load(f)
    world, agents = data["world"], data["agents"]
    sig_idx = [ACTIONS.index(f"signal_{i}") for i in range(1, N_SYMBOLS + 1)]
    rows = []  # (weight, situation_food, situation_energy, sym1 dist, sym2 dist)
    for a in agents:
        obs = a.sense(world)
        b = a.brain
        h = np.tanh(obs @ b.w1 + b.b1)  # memory is off in this stage: no carried state
        out = h @ b.w2 + b.b2
        p = np.exp(out - out.max())
        p /= p.sum()
        ps = p[sig_idx]
        w = ps.sum()
        if w <= 0:
            continue
        rows.append((w, food_dir(obs[:25]), int(a.energy > 5.0), ps / w, b.sym2_probs(h)))
    W = np.array([r[0] for r in rows])
    S1 = np.array([r[3] for r in rows])
    S2 = np.array([r[4] for r in rows])
    J = np.einsum("ni,nj->nij", S1, S2).reshape(len(rows), -1)  # joint (sym1,sym2), 16 cells
    rng = np.random.default_rng(0)

    def info(labels, k):
        onehot = np.eye(k)[labels] * W[:, None]
        return mi(onehot.T @ S1), mi(onehot.T @ J)

    print(f"{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}: "
          f"{len(agents)} agents, mean P(signal) {W.mean():.3f}")
    print(f"second-symbol usage (weighted): " +
          " ".join(f"{v:.2f}" for v in (W[:, None] * S2).sum(0) / W.sum()))
    out = {}
    for name, labels, k in [("food direction", np.array([r[1] for r in rows]), 6),
                            ("energy", np.array([r[2] for r in rows]), 2)]:
        i1, i12 = info(labels, k)
        sh = [info(rng.permutation(labels), k) for _ in range(50)]
        b1, b12 = np.mean([s[0] for s in sh]), np.mean([s[1] for s in sh])
        extra, extra_b = i12 - i1, b12 - b1
        print(f"  {name:<15} I(s;sym1) {i1:.4f} (shuffled {b1:.4f})   "
              f"I(s;sym2|sym1) {extra:.4f} (shuffled {extra_b:.4f})   "
              f"excess {extra - extra_b:+.4f} bits")
        out[name] = (i1 - b1, extra - extra_b)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
