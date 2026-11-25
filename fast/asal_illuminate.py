"""
ASAL illumination on Primordia: an atlas of the most different worlds the space allows.

ASAL's genetic algorithm (main_illuminate.py in github.com/SakanaAI/asal): keep an
archive of --archive worlds; each round, --children worlds are made by mutating random
archive members (Gaussian, --sigma), all are pictured, and then the least novel are
removed one at a time (smallest mean distance to their --k nearest neighbours, k = 2 as
in ASAL) until the archive is back to size. Each world is described by one vector: the
CLIP embedding of its last moment, both views (where creatures stand, their genes).

    python fast/asal_illuminate.py --out fast/runs/asal_atlas --rounds 60 --jobs 16

Resumable (archive.pkl). Output: atlas.png (last moment of every archive world, sorted
along the archive's main axis), atlas.csv (settings of every archive world), rounds.csv.
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import asal_core as AC
import asal_world as AW


def kill_least_novel(Z, n_kill, k):
    """ASAL's kill_least: repeatedly remove the member whose k nearest neighbours are closest."""
    D = -(Z @ Z.T)
    np.fill_diagonal(D, np.inf)
    killed = []
    for _ in range(n_kill):
        tki = int(np.sort(D, axis=-1)[:, :k].mean(axis=-1).argmin())
        D[:, tki] = np.inf
        D[tki, :] = np.inf
        killed.append(tki)
    return killed


def rollout_batch(pool, clip, U, seeds, a):
    res = pool.map(AW.rollout, [dict(kw=AW.decode(u), seed=int(s), ticks=a.ticks, every=a.every,
                                     pop_cap=a.pop_cap, min_pop=a.min_pop, burn_in=a.burn_in) for u, s in zip(U, seeds)], chunksize=1)
    out = []
    for r in res:
        z = AC.embed_run(clip, r)
        out.append(dict(z=AC.final_embedding(z), last=r["full"][-1], last_genes=r["gA"][-1],
                        status=r["status"], pop=r["pops"][-1], novelty=AC.score(z)[1]))
    return out


def save_atlas(arch, out):
    Z = np.stack([m["z"] for m in arch])
    c = Z - Z.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    order = np.argsort(c @ axis)
    n = len(arch)
    cols = int(np.ceil(np.sqrt(n)))
    h = arch[0]["last"].shape[0]
    tile = h // 2
    canvas = np.full(((tile + 2) * ((n + cols - 1) // cols), (tile + 2) * cols, 3), 255, np.uint8)
    for j, i in enumerate(order):
        im = arch[i]["last"][::2, ::2]
        r, q = divmod(j, cols)
        canvas[r * (tile + 2):r * (tile + 2) + tile, q * (tile + 2):q * (tile + 2) + tile] = im
    AC.save_png(canvas, os.path.join(out, "atlas.png"))
    with open(os.path.join(out, "atlas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "status", "final_pop", "novelty_space", "novelty_genes", "world"] + AW.NAMES)
        for j, i in enumerate(order):
            m = arch[i]
            w.writerow([j, m["status"], m["pop"], round(m["novelty"]["novelty_space"], 4),
                        round(m["novelty"]["novelty_genes"], 4), AW.describe(m["u"])] +
                       [round(float(v), 4) for v in m["u"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "asal_atlas"))
    ap.add_argument("--archive", type=int, default=64, help="ASAL uses 256 for cheap substrates")
    ap.add_argument("--children", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=0.1, help="ASAL default 0.1")
    ap.add_argument("--k", type=int, default=2, help="nearest neighbours (ASAL: 2 is best)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--pop-cap", type=int, default=12000)
    ap.add_argument("--min-pop", type=int, default=200)
    ap.add_argument("--burn-in", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-weights", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "archive.pkl")
    clip = AC.Clip(a.clip_weights, a.device)
    with AC.make_pool(a.jobs) as pool:
        if os.path.exists(path):
            with open(path, "rb") as f:
                st = pickle.load(f)
            arch, rnd = st["arch"], st["round"]
            rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
            print(f"resuming at round {rnd}")
        else:
            with open(os.path.join(a.out, "config.json"), "w") as f:
                json.dump(vars(a), f, indent=1)
            rng = np.random.default_rng(a.seed)
            # first archive: spread over the whole space (Latin hypercube), not all at one point
            U = (np.stack([rng.permutation(a.archive) for _ in range(AW.D)], 1) + rng.random((a.archive, AW.D))) / a.archive
            t0 = time.time()
            got = rollout_batch(pool, clip, U, rng.integers(1, 10 ** 6, a.archive), a)
            arch = [dict(g, u=u) for g, u in zip(got, U)]
            rnd = 0
            print(f"first archive of {a.archive} worlds pictured in {time.time() - t0:.0f} s")
        log = os.path.join(a.out, "rounds.csv")
        if not os.path.exists(log):
            with open(log, "w", newline="") as f:
                csv.writer(f).writerow(["round", "illumination", "healthy_in_archive", "children_kept", "secs"])
        while rnd < a.rounds:
            t0 = time.time()
            par = rng.integers(0, len(arch), a.children)
            U = np.clip(np.stack([arch[p]["u"] for p in par]) + a.sigma * rng.normal(size=(a.children, AW.D)), 0, 1)
            kids = [dict(g, u=u) for g, u in zip(rollout_batch(pool, clip, U, rng.integers(1, 10 ** 6, a.children), a), U)]
            pool_all = arch + kids
            killed = set(kill_least_novel(np.stack([m["z"] for m in pool_all]), a.children, a.k))
            arch = [m for i, m in enumerate(pool_all) if i not in killed]
            kept = sum(1 for i in range(len(pool_all) - a.children, len(pool_all)) if i not in killed)
            rnd += 1
            ill = AC.asal_illumination(np.stack([m["z"] for m in arch]))
            healthy = sum(m["status"] == "ok" for m in arch)
            with open(log, "a", newline="") as f:
                csv.writer(f).writerow([rnd, round(ill, 5), healthy, kept, round(time.time() - t0)])
            with open(path + ".tmp", "wb") as f:
                pickle.dump(dict(arch=arch, round=rnd, rng=rng.bit_generator.state), f)
            os.replace(path + ".tmp", path)
            if rnd % 5 == 0 or rnd == a.rounds:
                save_atlas(arch, a.out)
            print(f"round {rnd}/{a.rounds}  illumination {ill:.4f} (lower = more varied)  "
                  f"healthy worlds {healthy}/{len(arch)}  new kept {kept}/{a.children}  {time.time() - t0:.0f} s", flush=True)
    save_atlas(arch, a.out)
    print("done; atlas in", os.path.join(a.out, "atlas.png"))


if __name__ == "__main__":
    main()
