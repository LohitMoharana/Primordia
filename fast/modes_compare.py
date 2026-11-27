"""
Compare worlds with MODES (oee_modes.py), the published open-ended-evolution metrics.

Each world runs on fresh seeds; MODES change / novelty / diversity / complexity are measured
every 500 ticks with a 5000-tick persistence filter. Worlds:

    start                 the healthy default world (asal_world.default_start)
    mode:<name>:<on|off>  a world from modes.py (e.g. mode:open_brains:on), stage-5 founders
    best:<dir>            the best world of an ASAL search (reads <dir>/best.txt)
    u:[0.1, 0.5, ...]     any point of the search space

    python fast/modes_compare.py --worlds start best:fast/runs/asal_open3 --seeds 10 --ticks 100000 --jobs 16

Output (--out): runs.csv (one line per run), timeline.csv (every snapshot), summary printed
with a permutation test between the first two worlds. Each run is saved when it finishes,
so a stopped comparison resumes.
"""
import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import asal_world as AW
import oee_modes as M

# sensitivity check (--variants): name, skeleton, persistence interval
VARIANTS = [("i2500", "strategy", 2500), ("i10000", "strategy", 10000),
            ("coarse", "coarse", 5000), ("fine", "fine", 5000)]

METRICS = ["change", "novelty", "diversity", "complexity",
           "units_mean", "units_max", "units_persistent",   # brain units: worlds with evolving brain size only
           "macros_owned", "macro_len_mean", "macro_len_max", "macro_share",  # evolvable actions only
           "scavengers", "gut_mean", "dig_rate", "remains_left",  # remains worlds only
           "pop_mean", "novelty_per_1k"]  # mean population, and novelty per 1000 creatures
METRICS += [f"{m}_{v[0]}" for v in VARIANTS for m in ("novelty", "per1k", "diversity")]  # --variants only


def parse_mods(spec):
    """'<world>|k=v,k=v': changes to a world. scale=2 (asal_world.scale_world), shake_every=1
    (a shaken world), or any setting, e.g. catch_p=0.2."""
    if "|" not in spec:
        return spec, {}
    base, m = spec.split("|", 1)
    mods = {}
    for kv in m.split(","):
        k, v = kv.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        mods[k.strip()] = v
    return base, mods


def apply_mods(kw, mods):
    kw = dict(kw)
    for k, v in mods.items():
        if k == "scale":
            kw = AW.scale_world(kw, v)
        else:
            kw[k] = v
    return kw


def make_world(spec, seed):
    """The simulation for a world spec (same class choice as run_fast.py for modes)."""
    spec, mods = parse_mods(spec)
    if spec.startswith("mode:") and mods:
        raise SystemExit("world changes (|...) are for start / best: / u: worlds")
    if mods:
        return AW.make_sim(apply_mods(AW.decode(world_u(spec)), mods), seed)
    if spec.startswith("mode:"):
        _, name, arm = spec.split(":")
        from modes import MODES
        kw = MODES[name]["kw"](arm == "on")
        F = AW._founders(seed)
        if "remains_source" in kw:
            from eco import EcoSim as cls
        elif "actions_evolve" in kw:
            from actions import PredActionSim, WaysActionSim
            cls = WaysActionSim if "buried" in kw else PredActionSim
        elif "brain_evolve" in kw:
            from brains import PredBrainSim, WaysBrainSim
            cls = WaysBrainSim if "buried" in kw else PredBrainSim
        elif "buried" in kw:
            from ways import WaysSim as cls
        elif "predators" in kw:
            from predators import PredSim as cls
        else:
            from engine import FastSim as cls
        return cls(seed=seed, founders=F, **kw)
    return AW.make_sim(AW.decode(world_u(spec)), seed)


def describe(spec):
    base, mods = parse_mods(spec)
    if base.startswith("mode:"):
        return spec
    return AW.describe(world_u(base)) + ("" if not mods else " | " + ", ".join(f"{k}={v}" for k, v in mods.items()))


def world_u(spec):
    if spec == "start":
        return AW.default_start()
    if spec.startswith("best:"):
        lines = open(os.path.join(spec[5:], "best.txt")).read().splitlines()
        return np.array(json.loads(lines[-1]))
    if spec.startswith("u:"):
        return np.array(json.loads(spec[2:]))
    raise SystemExit(f"unknown world {spec}")


def run_one(job):
    AW.worker_init()
    t0 = time.time()
    sim = make_world(job["world"], job["seed"])
    P = M.load_probes()
    T = M.ModesTracker(P, job["resolution"], job["interval"], prune_tree=not job.get("variants"))
    T.attach(sim)
    extra = {}
    if job.get("variants"):  # sensitivity check: the same run measured other ways
        for name, sk, iv in VARIANTS:
            extra[name] = M.ModesTracker(P, job["resolution"], iv, skeleton=sk, share_tree_with=T)
    bedau = None
    if job.get("bedau"):
        from oee_bedau import BedauTracker
        bedau = BedauTracker(P, T, job["resolution"], seed=job["seed"])
    status = "ok"
    while sim.tick < job["ticks"]:
        sim.step()
        T.update(sim)
        for t in extra.values():
            t.update(sim)
        if bedau is not None:
            bedau.update(sim)
        if sim.population == 0:
            status = f"extinct@{sim.tick}"
            break
        if sim.population > job["pop_cap"]:
            status = f"exploded@{sim.tick}"
            break
    s = T.summary()
    for name, t in extra.items():
        v = t.summary()
        s.update({f"novelty_{name}": v["novelty"], f"per1k_{name}": v["novelty_per_1k"],
                  f"diversity_{name}": v["diversity"]})
    if bedau is not None:
        os.makedirs(os.path.join(job["out"], "bedau"), exist_ok=True)
        slug = "".join(ch if ch.isalnum() else "_" for ch in job["world"])[:60]
        bedau.save(os.path.join(job["out"], "bedau", f"{slug}_{job['seed']}.npz"), world=job["world"], seed=job["seed"])
    return dict(world=job["world"], seed=job["seed"], status=status, final_pop=sim.population,
                secs=round(time.time() - t0), rows=T.rows, **s)


def perm_test(a, b, n=20000, seed=0):
    """Two-sided permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    x = np.concatenate([a, b])
    r = np.random.default_rng(seed)
    hits = sum(abs((p := r.permutation(x))[:len(a)].mean() - p[len(a):].mean()) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", default=["start"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1001, help="fresh seeds, never used by the searches")
    ap.add_argument("--ticks", type=int, default=100000)
    ap.add_argument("--resolution", type=int, default=500)
    ap.add_argument("--interval", type=int, default=5000, help="persistence filter (about 25 generations)")
    ap.add_argument("--pop-cap", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "modes_compare"))
    ap.add_argument("--variants", action="store_true", help="also measure with 2 other persistence windows and 2 other skeletons")
    ap.add_argument("--bedau", action="store_true", help="also record Bedau's activity statistics with a neutral shadow")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    M.load_probes()  # make the probe file once, before the workers start
    runs_path, tl_path = os.path.join(a.out, "runs.csv"), os.path.join(a.out, "timeline.csv")
    done = set()
    if os.path.exists(runs_path):
        done = {(r["world"], int(r["seed"])) for r in csv.DictReader(open(runs_path))}
    else:
        with open(runs_path, "w", newline="") as f:
            csv.writer(f).writerow(["world", "seed", "status", "final_pop", "secs", "snapshots"] + METRICS + ["settings"])
        with open(tl_path, "w", newline="") as f:
            csv.writer(f).writerow(["world", "seed", "tick", "change", "novelty", "diversity", "complexity", "filter_on"])
    jobs = []
    for w in a.worlds:
        for s in range(a.first_seed, a.first_seed + a.seeds):
            if (w, s) not in done:
                jobs.append(dict(world=w, seed=s, ticks=a.ticks, resolution=a.resolution, out=a.out,
                                 variants=a.variants, bedau=a.bedau,
                                 interval=a.interval, pop_cap=a.pop_cap))
    print(f"{len(jobs)} runs to do ({len(done)} already done); {a.jobs} at a time", flush=True)
    for w in a.worlds:
        print(f"  {w}: {describe(w)}")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.jobs, initializer=AW.worker_init) as pool:
        for r in pool.imap_unordered(run_one, jobs):
            with open(runs_path, "a", newline="") as f:
                csv.writer(f).writerow([r["world"], r["seed"], r["status"], r["final_pop"], r["secs"], r["snapshots"]] +
                                       [round(float(r.get(m, 0)), 4) for m in METRICS] + [describe(r["world"])])
            with open(tl_path, "a", newline="") as f:
                csv.writer(f).writerows([[r["world"], r["seed"], x[0], x[1], x[2], round(x[3], 4), x[4], int(x[5])]
                                         for x in r["rows"]])
            print(f"{r['world']} seed {r['seed']}: {r['status']}, {r['final_pop']} creatures, "
                  + ", ".join(f"{m} {float(r.get(m, 0)):.2f}" for m in METRICS) + f"  ({r['secs']} s)", flush=True)
    rows = list(csv.DictReader(open(runs_path)))
    print("\nworld                 runs  survived   " + "   ".join(f"{m:>16s}" for m in METRICS))
    print("(units_*: hidden units in use at the end - everyone, max, and roots of persistent lineages)")
    by = {}
    for w in a.worlds:
        rs = [r for r in rows if r["world"] == w]
        by[w] = rs
        ok = sum(r["status"] == "ok" for r in rs)
        cells = []
        for m in METRICS:
            v = np.array([float(r[m]) for r in rs])
            cells.append(f"{v.mean():7.2f} +- {v.std(ddof=1) if len(v) > 1 else 0:5.2f}")
        print(f"{w[:20]:20s}  {len(rs):4d}  {ok:4d}/{len(rs):<4d}  " + "   ".join(cells))
    if len(a.worlds) >= 2 and all(len(by[w]) > 1 for w in a.worlds[:2]):
        w1, w2 = a.worlds[:2]
        print(f"\npermutation test, {w1} vs {w2} (two-sided p):")
        for m in METRICS:
            p = perm_test([float(r[m]) for r in by[w1]], [float(r[m]) for r in by[w2]])
            print(f"  {m:10s} p = {p:.4f}")
        print("(dead worlds count with whatever they reached before dying; see 'survived')")


if __name__ == "__main__":
    main()
