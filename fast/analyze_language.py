"""
Does the second symbol carry information? (fast-engine checkpoints)

    python3 analyze_language.py <checkpoint.pkl> [<checkpoint.pkl> ...]

For every agent, in its actual situation, compute from its brain (no sampling
noise) how likely it is to signal, its distribution over first symbols and its
distribution over second symbols. Situations:
  food here     : food in its own cell vs none
  energy        : below or above half
  cluster dir   : (sparse world) which of 8 directions the nearest food-cluster
                  centre lies in, often beyond sensing range
Reports, weighted by each agent's chance of signalling, in bits:
  I(situation; symbol 1) and the extra the second symbol adds,
  I(situation; symbol 2 | symbol 1), each minus a shuffled-label baseline.
Compare a language-ON run with its OFF twin: in OFF the second symbol is never
delivered, so whatever structure it has there is drift.
"""
import sys

import numpy as np

import engine as E
from engine import FastSim


def mi(j):
    j = j / j.sum()
    a, b = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log2(j[nz] / (a @ b)[nz])).sum())


def analyse(path, seed=0):
    sim = FastSim.from_checkpoint(path)
    S = sim.size
    idx = np.flatnonzero(sim.alive)
    occ = np.zeros((S, S)); np.add.at(occ, (sim.x[idx], sim.y[idx]), 1)
    sig = np.zeros((S, S)); sig2 = np.zeros((S, S))
    la = sim.last_action[idx]
    s = (la >= E.SIG0) & (la < E.SIG0 + E.N_SYMBOLS)
    if sim.cfg["communication"]:
        sig[sim.x[idx][s], sim.y[idx][s]] = la[s] - E.SIG0 + 1
        if sim.cfg.get("language", False):
            sig2[sim.x[idx][s], sim.y[idx][s]] = sim.last_sym2[idx][s]
    X = sim.sense(idx, occ, sig, sig2)
    pre = np.einsum("ni,nih->nh", X, sim.w1[idx]) + sim.b1[idx]
    if sim.cfg.get("memory", False):
        pre += np.einsum("ni,nij->nj", sim.hstate[idx], sim.wh[idx])
    h = np.tanh(pre)
    out = np.einsum("nh,nho->no", h, sim.w2[idx]) + sim.b2[idx]
    p = np.exp(out - out.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    ps = p[:, E.SIG0:E.SIG0 + E.N_SYMBOLS]
    w = ps.sum(1)
    S1 = ps / w[:, None]
    lg = np.einsum("nh,nhq->nq", h, sim.ws[idx]) + sim.bs[idx]
    S2 = np.exp(lg - lg.max(1, keepdims=True)); S2 /= S2.sum(1, keepdims=True)
    J = np.einsum("ni,nj->nij", S1, S2).reshape(len(idx), -1)
    labels = {"food here": (sim.food[sim.x[idx], sim.y[idx]] > 0.05).astype(int),
              "energy": (sim.energy[idx] > 5.0).astype(int)}
    if hasattr(sim, "patch_xy") and getattr(sim, "patch_xy", None) is not None:
        dx = sim.patch_xy[None, :, 0] - sim.x[idx, None]; dx = (dx + S / 2) % S - S / 2
        dy = sim.patch_xy[None, :, 1] - sim.y[idx, None]; dy = (dy + S / 2) % S - S / 2
        k = np.argmin(dx ** 2 + dy ** 2, 1)
        ang = np.arctan2(dy[np.arange(len(idx)), k], dx[np.arange(len(idx)), k])
        labels["cluster dir"] = ((ang + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    rng = np.random.default_rng(seed)
    res = {}
    for name, lab in labels.items():
        kk = lab.max() + 1
        def info(L):
            oh = np.eye(kk)[L] * w[:, None]
            return mi(oh.T @ S1), mi(oh.T @ J)
        i1, i12 = info(lab)
        sh = [info(rng.permutation(lab)) for _ in range(30)]
        b1, b12 = np.mean([x[0] for x in sh]), np.mean([x[1] for x in sh])
        res[name] = (i1 - b1, (i12 - i1) - (b12 - b1))
    return dict(pop=len(idx), p_signal=float(w.mean()), usage=(w[:, None] * S2).sum(0) / w.sum(), info=res)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = analyse(path)
        print(f"{path}: {r['pop']} agents, mean P(signal) {r['p_signal']:.3f}, "
              f"2nd-symbol use {' '.join(f'{x:.2f}' for x in r['usage'])}")
        for k, (a, b) in r["info"].items():
            print(f"   {k:<12} symbol 1: {a:+.4f} bits   symbol 2 adds: {b:+.4f} bits (excess over shuffled)")
