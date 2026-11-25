"""
ASAL search on Primordia: find world settings (asal_world.SPACE) with Sep-CMA-ES, as ASAL does.

  open-endedness (default): worlds whose creatures keep producing pictures unlike any
      earlier picture, measured with the noise floor (asal_core.novelty)
  target: worlds that look like a sequence of prompts over the run, ASAL's supervised
      target ("a;b;c" = like a early, b in the middle, c late)

    python fast/asal_search.py --out fast/runs/asal_open --gens 80 --jobs 16
    python fast/asal_search.py --objective target --prompts "scattered dots;two separate colonies" --out fast/runs/asal_two
    python fast/asal_search.py --selftest

Resumable (search.pkl saved every generation). Output: evals.csv (every world tried),
best.txt, best_strip.png (top: where creatures stand; bottom: their genes), best.gif.
See asal_core.py for what comes from ASAL / ASAL++ and what is new here.
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


def add_args(ap):
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--popsize", type=int, default=16, help="ASAL default 16")
    ap.add_argument("--bs", type=int, default=1, help="world seeds per setting, averaged (ASAL's bs)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--pop-cap", type=int, default=12000)
    ap.add_argument("--min-pop", type=int, default=200, help="below this a world counts as collapsed (2 x 100 drawn)")
    ap.add_argument("--burn-in", type=int, default=3000, help="settling ticks: only extinction counts, no pictures")
    ap.add_argument("--sigma0", type=float, default=0.2)
    ap.add_argument("--full-cma", action="store_true", help="full CMA-ES instead of ASAL's Sep-CMA-ES")
    ap.add_argument("--seed", type=int, default=0, help="search seed (world seeds are drawn from it)")
    ap.add_argument("--clip-weights", default=None, help="local CLIP checkpoint (else downloaded once)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--coef-prompt", type=float, default=1.0)
    ap.add_argument("--coef-softmax", type=float, default=0.0)
    ap.add_argument("--coef-oe", type=float, default=0.0, help="target objective: add this much open-endedness")


def run_search(a, out, clip, pool, objective="open", prompts=None, x0=None, log=print):
    """One CMA search (resumable from out/search.pkl). Returns the best dict (u, loss, ...)."""
    import cma
    os.makedirs(out, exist_ok=True)
    state_path = os.path.join(out, "search.pkl")
    z_txt = clip.texts(prompts) if objective == "target" else None
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        es, gen, best = st["es"], st["gen"], st["best"]
        rng = np.random.default_rng(); rng.bit_generator.state = st["rng"]
        log(f"resuming {out} at generation {gen}")
    else:
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(dict(vars(a), objective=objective, prompts=prompts,
                           x0=None if x0 is None else [float(v) for v in x0]), f, indent=1)
        opts = dict(bounds=[0, 1], popsize=a.popsize, seed=a.seed + 1, verbose=-9)
        if not a.full_cma:
            opts["CMA_diagonal"] = True  # Sep-CMA-ES, as in ASAL
        es = cma.CMAEvolutionStrategy(AW.default_start() if x0 is None else np.asarray(x0, float), a.sigma0, opts)
        gen, best = 0, None
        rng = np.random.default_rng(a.seed)
    ev_path = os.path.join(out, "evals.csv")
    part_keys = ["novelty_space", "novelty_genes", "asal_oe"] + (["target", "softmax"] if objective == "target" else [])
    if not os.path.exists(ev_path):
        with open(ev_path, "w", newline="") as f:
            csv.writer(f).writerow(["gen", "i", "world_seeds", "loss"] + part_keys +
                                   ["status", "final_pop", "max_gen", "secs"] + AW.NAMES)
    score_kw = dict(objective=objective, z_txt=z_txt, coef_prompt=a.coef_prompt,
                    coef_softmax=a.coef_softmax, coef_oe=a.coef_oe)
    while gen < a.gens:
        t0 = time.time()
        U = es.ask()
        seeds = [int(s) for s in rng.integers(1, 10 ** 6, a.bs)]  # same seeds for the whole generation
        losses, infos, firsts = AC.evaluate(pool, clip, U, seeds, a.ticks, a.every, a.pop_cap, a.min_pop,
                                               getattr(a, "burn_in", 3000), **score_kw)
        rows = []
        for i, (u, l, inf) in enumerate(zip(U, losses, infos)):
            rows.append([gen, i, " ".join(map(str, seeds)), round(l, 5)] + [round(inf[k], 5) for k in part_keys] +
                        [inf["status"], inf["final_pop"], inf["max_gen"], round(inf["secs"], 1)] +
                        [round(float(v), 4) for v in u])
            if best is None or l < best["loss"]:
                r, _ = firsts[i]
                best = dict(loss=l, u=np.array(u), gen=gen, seeds=seeds, info=inf)
                AC.save_png(AC.strip(r["full"], 8, r["gA"]), os.path.join(out, "best_strip.png"))
                AC.save_gif(r["full"], os.path.join(out, "best.gif"))
        es.tell(U, losses)
        with open(ev_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        gen += 1
        with open(state_path + ".tmp", "wb") as f:
            pickle.dump(dict(es=es, gen=gen, best=best, rng=rng.bit_generator.state), f)
        os.replace(state_path + ".tmp", state_path)
        with open(os.path.join(out, "best.txt"), "w") as f:
            f.write(f"loss {best['loss']:.5f} (generation {best['gen']}, world seeds {best['seeds']}, "
                    f"{best['info']['status']}, {best['info']['final_pop']} creatures at the end)\n")
            f.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best["info"].items()}) + "\n")
            f.write(AW.describe(best["u"]) + "\n")
            f.write(json.dumps([round(float(v), 4) for v in best["u"]]) + "\n")
        alive = sum(inf["status"].count("ok") == a.bs for inf in infos)
        log(f"gen {gen}/{a.gens}  best {min(losses):.4f}  median {np.median(losses):.4f}  "
            f"healthy {alive}/{len(U)}  all-time {best['loss']:.4f}  {time.time() - t0:.0f} s")
        if gen % 10 == 0 or gen == a.gens:
            log("   search centre: " + AW.describe(es.result.xfavorite))
    return best


def selftest(a):
    t = time.time()
    clip = AC.Clip(a.clip_weights, a.device)
    print(f"CLIP loaded on {clip.device} in {time.time() - t:.0f} s")
    AW.worker_init()
    r = AW.rollout(dict(kw=AW.decode(AW.default_start()), seed=1, ticks=6000, every=500))
    z = AC.embed_run(clip, r)
    l, p = AC.score(z)
    print(f"world ran 6000 ticks in {r['secs']:.0f} s ({r['status']}), {r['pops'][-1]} creatures; "
          f"novelty {-l:.4f} (space {p['novelty_space']:.4f}, genes {p['novelty_genes']:.4f}); "
          f"ASAL's original score {p['asal_oe']:.3f}")
    zt = clip.texts(["coloured dots on a black background", "a photo of a cat"])
    print("similarity to 'coloured dots on a black background' vs 'a photo of a cat':",
          np.round((z["full"] @ zt.T).mean(0), 3))
    os.makedirs(a.out, exist_ok=True)
    AC.save_png(AC.strip(r["full"], 6, r["gA"]), os.path.join(a.out, "selftest.png"))
    print("selftest ok; picture saved to", os.path.join(a.out, "selftest.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "asal_open"))
    ap.add_argument("--objective", choices=["open", "target"], default="open")
    ap.add_argument("--prompts", default=None, help="target prompts over the run, separated by ';'")
    ap.add_argument("--x0", default=None, help="start the search here (JSON list, e.g. from a best.txt)")
    ap.add_argument("--selftest", action="store_true")
    add_args(ap)
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    if a.objective == "target" and not a.prompts:
        sys.exit("--objective target needs --prompts")
    prompts = [p.strip() for p in a.prompts.split(";")] if a.prompts else None
    clip = AC.Clip(a.clip_weights, a.device)
    print(f"CLIP on {clip.device}; {a.jobs} worlds at a time; {a.popsize} settings x {a.bs} seeds per generation")
    with AC.make_pool(a.jobs) as pool:
        best = run_search(a, a.out, clip, pool, a.objective, prompts, json.loads(a.x0) if a.x0 else None,
                          log=lambda s: print(s, flush=True))
    print("done. best world:", AW.describe(best["u"]))


if __name__ == "__main__":
    main()
