"""
ASAL's search, scored with MODES: find the kind of world (asal_world.SPACE) whose creatures keep
producing lasting new strategies.

ASAL (Kumar et al. 2024) searches simulation settings with Sep-CMA-ES and scores runs with an image
model. On Primordia the image score picked a world that dies out in 9/10 runs (FINDINGS.md). Here
the search is the same, but the score is the field's own measure of open-ended evolution, MODES
(Dolson et al. 2019; oee_modes.py): the number of new strategies that pass the persistence filter
(still have living descendants 5,000 ticks later) over the run. A world that dies stops scoring.

    python fast/modes_search.py --out fast/runs/modes_search --gens 24 --jobs 16 [--open-brains]

--open-brains: creatures in every world have brains without a fixed size (Soros & Stanley's
condition 4; brains.py), with working memory on.
Resumable (search.pkl after every generation). Output: evals.csv, best.txt.
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

import asal_world as AW
import oee_modes as M

BRAIN = dict(memory=True, zero_memory_weights=True, brain_evolve=True, brain_open=True, nact_mut_p=0.1,
             brain_wired=True)


def run_world(job):
    AW.worker_init()
    t0 = time.time()
    kw = AW.decode(np.array(job["u"]))
    if job["open_brains"]:
        from brains import WaysBrainSim
        sim = WaysBrainSim(seed=job["seed"], founders=AW._founders(job["seed"]), **dict(kw, **BRAIN))
    else:
        sim = AW.make_sim(kw, job["seed"])
    T = M.ModesTracker(M.load_probes(), 500, job["interval"])
    T.attach(sim)
    status = "ok"
    while sim.tick < job["ticks"]:
        sim.step()
        T.update(sim)
        if sim.population == 0:
            status = f"extinct@{sim.tick}"
            break
        if sim.population > job["pop_cap"]:
            status = f"exploded@{sim.tick}"  # stops scoring here (and costs no more time)
            break
    s = T.summary()
    return dict(s, status=status, final_pop=sim.population, secs=time.time() - t0,
                units_mean=s.get("units_mean", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "modes_search"))
    ap.add_argument("--gens", type=int, default=24)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ticks", type=int, default=40000)
    ap.add_argument("--interval", type=int, default=5000, help="MODES persistence filter (ticks)")
    ap.add_argument("--pop-cap", type=int, default=12000)
    ap.add_argument("--sigma0", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--open-brains", action="store_true")
    a = ap.parse_args()
    import cma
    os.makedirs(a.out, exist_ok=True)
    M.load_probes()
    sp = os.path.join(a.out, "search.pkl")
    if os.path.exists(sp):
        with open(sp, "rb") as f:
            st = pickle.load(f)
        es, gen, best = st["es"], st["gen"], st["best"]
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        print(f"resuming at generation {gen}", flush=True)
    else:
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(vars(a), f, indent=1)
        es = cma.CMAEvolutionStrategy(AW.default_start(), a.sigma0,
                                      dict(bounds=[0, 1], popsize=a.popsize, seed=a.seed + 1, verbose=-9,
                                           CMA_diagonal=True))
        gen, best = 0, None
        rng = np.random.default_rng(a.seed)
    ev = os.path.join(a.out, "evals.csv")
    if not os.path.exists(ev):
        with open(ev, "w", newline="") as f:
            csv.writer(f).writerow(["gen", "i", "world_seed", "novelty", "change", "diversity", "complexity",
                                    "units_mean", "status", "final_pop", "secs"] + AW.NAMES)
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.jobs, initializer=AW.worker_init) as pool:
        while gen < a.gens:
            t0 = time.time()
            U = es.ask()
            seed = int(rng.integers(1, 10 ** 6))
            res = pool.map(run_world, [dict(u=[float(v) for v in u], seed=seed, ticks=a.ticks, interval=a.interval,
                                             pop_cap=a.pop_cap, open_brains=a.open_brains) for u in U], chunksize=1)
            losses = [-r["novelty"] for r in res]
            with open(ev, "a", newline="") as f:
                w = csv.writer(f)
                for i, (u, r) in enumerate(zip(U, res)):
                    w.writerow([gen, i, seed, r["novelty"], round(r["change"], 3), round(r["diversity"], 3),
                                r["complexity"], round(r["units_mean"], 2), r["status"], r["final_pop"],
                                round(r["secs"])] + [round(float(v), 4) for v in u])
                    if best is None or -r["novelty"] < best["loss"]:
                        best = dict(loss=-r["novelty"], u=np.array(u), gen=gen, seed=seed, res=r)
            es.tell(U, losses)
            gen += 1
            with open(sp + ".tmp", "wb") as f:
                pickle.dump(dict(es=es, gen=gen, best=best, rng=rng.bit_generator.state), f)
            os.replace(sp + ".tmp", sp)
            with open(os.path.join(a.out, "best.txt"), "w") as f:
                r = best["res"]
                f.write(f"novelty {r['novelty']:.0f}, change {r['change']:.2f}, diversity {r['diversity']:.2f}, "
                        f"{r['status']}, {r['final_pop']} creatures (generation {best['gen']}, world seed {best['seed']})\n")
                f.write(AW.describe(best["u"]) + "\n")
                f.write(json.dumps([round(float(v), 4) for v in best["u"]]) + "\n")
            ok = sum(r["status"] == "ok" for r in res)
            print(f"gen {gen}/{a.gens}  best novelty {-min(losses):.0f}  median {-np.median(losses):.0f}  "
                  f"survived {ok}/{len(res)}  all-time {-best['loss']:.0f}  {(time.time() - t0) / 60:.0f} min", flush=True)
            if gen % 4 == 0:
                print("   search centre:", AW.describe(es.result.xfavorite), flush=True)


if __name__ == "__main__":
    main()
