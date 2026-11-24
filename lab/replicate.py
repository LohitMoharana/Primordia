"""
Run many Primordia experiments at once (one process per run), on any OS.

Quick start (from the bundle folder):

    python lab/replicate.py --preset kept --seeds 2 3 4 5        # re-test the 4 kept features
    python lab/replicate.py --preset rejected --seeds 2 3        # re-test the rejected ones
    python lab/replicate.py --stages 11_selfpred --seeds 1       # finish the cloud's stage 11 run
    python lab/replicate.py --preset all --seeds 2 3 4 5 --jobs 14

Each (stage, arm, seed) run writes to stages/<stage>/runs/<prefix>_<arm>_seed<N>/.
Safe to stop (Ctrl+C) and rerun the same command: finished runs (DONE) and
extinct runs (EXTINCT) are skipped, interrupted runs resume bit-exactly from
their latest checkpoint. Afterwards:  python lab/summarize.py
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # the same Python you started this with (python / python3 / py)

# stage -> (run-dir prefix, launcher script + args before arm/seed, args after, target ticks)
STAGES = {
    "01_social_sensing": ("ablation",  ["run_ablation.py"], [], 20000),
    "02_communication":  ("comm",      ["run_comm_ablation.py"], [], 20000),
    "03_seasons":        ("seasons",   ["run_stage.py"], ["--communication", "1"], 25000),
    "04_memory":         ("memory",    ["run_stage.py"], ["--seasons", "1"], 25000),
    "05_hazards":        ("hazards",   ["run_stage.py"], [], 25000),
    "06_kin":            ("kin",       ["run_stage.py"], [], 25000),
    "06b_kin":           ("kin",       ["run_stage.py"], [], 25000),
    "07_sexual":         ("sexual",    ["run_stage.py"], [], 25000),
    "08_predators":      ("predators", ["run_stage.py"], [], 25000),
    "09_markers":        ("markers",   ["run_stage.py"], [], 25000),
    "10_learning":       ("learning",  ["run_stage.py"], [], 25000),
    "11_selfpred":       ("selfpred",  ["run_stage.py"], [], 25000),
    "12_language":       ("language",  ["run_stage.py"], [], 25000),
    "13_cofood":         ("cofood",    ["run_stage.py"], [], 25000),
}
PRESETS = {
    "kept":     ["01_social_sensing", "02_communication", "03_seasons", "05_hazards"],
    "rejected": ["04_memory", "06b_kin", "07_sexual", "08_predators", "09_markers", "10_learning", "11_selfpred", "12_language"],
    "promising": ["13_cofood"],
    "all":      sorted(STAGES),
}


def latest_ckpt(run_dir):
    ck = []
    for p in glob.glob(os.path.join(run_dir, "checkpoint_*.pkl")):
        m = re.search(r"checkpoint_(\d+)\.pkl$", p)
        if m:
            ck.append((int(m.group(1)), p))
    return max(ck) if ck else None


def is_empty(path):
    """True if a checkpoint holds no agents (written when a run goes extinct).
    Only small files are opened: a checkpoint with agents is many megabytes."""
    if os.path.getsize(path) > 5_000_000:
        return False
    import pickle

    class _Any:  # stands in for the stage's own classes; only the agent count matters
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _Loader(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("numpy", "builtins", "collections", "_codecs", "copyreg"):
                return super().find_class(module, name)
            return _Any

    try:
        with open(path, "rb") as f:
            return len(_Loader(f).load().get("agents", [1])) == 0
    except Exception:
        return False


def last_pop(run_dir):
    p = os.path.join(run_dir, "log.csv")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["population"]) if rows else None


def command(stage, arm, seed):
    prefix, pre, post, target = STAGES[stage]
    code = os.path.join(ROOT, "stages", stage)
    run_rel = os.path.join("runs", f"{prefix}_{arm}_seed{seed}")
    run_dir = os.path.join(code, run_rel)
    if os.path.exists(os.path.join(run_dir, "DONE")) or os.path.exists(os.path.join(run_dir, "EXTINCT")):
        return None, code, run_dir, target
    ck = latest_ckpt(run_dir)
    if ck and ck[0] < target and is_empty(ck[1]):
        # the run died out: its final checkpoint holds no agents
        open(os.path.join(run_dir, "EXTINCT"), "w").close()
        return None, code, run_dir, target
    if ck:
        tick, path = ck
        if tick >= target:
            open(os.path.join(run_dir, "DONE"), "w").close()
            return None, code, run_dir, target
        return [PY, "resume.py", path, run_rel, "--ticks", str(target - tick)], code, run_dir, target
    if os.path.exists(os.path.join(run_dir, "log.csv")):
        # a copied cloud log with no checkpoint: keep it, don't overwrite
        return "keep", code, run_dir, target
    return [PY] + pre + [arm, str(seed)] + post, code, run_dir, target


def progress(run_dir):
    p = os.path.join(run_dir, "log.csv")
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 400))
            last = f.read().decode(errors="ignore").strip().splitlines()[-1].split(",")
        return int(float(last[0])), int(float(last[1]))
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=sorted(STAGES), default=[])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--arms", nargs="+", choices=["on", "off"], default=["on", "off"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="runs at once (default: CPU threads minus 2)")
    a = ap.parse_args()
    stages = list(dict.fromkeys(a.stages + (PRESETS[a.preset] if a.preset else [])))
    if not stages:
        ap.error("give --stages and/or --preset")

    queue = [(s, arm, seed) for seed in a.seeds for s in stages for arm in a.arms]
    running = []
    print(f"{len(queue)} runs queued, up to {a.jobs} at once (Python: {PY})")
    last_report = 0
    try:
        while queue or running:
            while queue and len(running) < a.jobs:
                s, arm, seed = queue.pop(0)
                cmd, code, run_dir, target = command(s, arm, seed)
                if cmd is None:
                    print(f"skip  {s} {arm} seed{seed} (done or extinct)")
                    continue
                if cmd == "keep":
                    print(f"skip  {s} {arm} seed{seed} (cloud log only, no checkpoint; use a new seed)")
                    continue
                os.makedirs(run_dir, exist_ok=True)
                p = subprocess.Popen(cmd, cwd=code, stdout=subprocess.DEVNULL,
                                     stderr=open(os.path.join(run_dir, "stderr.log"), "a"))
                running.append((p, s, arm, seed, run_dir, target))
                print(f"start {s} {arm} seed{seed}")
            for item in list(running):
                p, s, arm, seed, run_dir, target = item
                if p.poll() is None:
                    continue
                running.remove(item)
                ck = latest_ckpt(run_dir)
                if last_pop(run_dir) == 0 or (ck and is_empty(ck[1])):
                    open(os.path.join(run_dir, "EXTINCT"), "w").close()
                    print(f"EXTINCT {s} {arm} seed{seed}")
                elif p.returncode == 0 and ck and ck[0] >= target:
                    open(os.path.join(run_dir, "DONE"), "w").close()
                    print(f"done  {s} {arm} seed{seed}")
                else:
                    print(f"stopped {s} {arm} seed{seed} (code {p.returncode}); see stderr.log, rerun to resume")
            if time.time() - last_report > 300 and running:
                last_report = time.time()
                rows = []
                for p, s, arm, seed, run_dir, target in running:
                    t, pop = progress(run_dir)
                    rows.append(f"{s}/{arm}/s{seed}: tick {t}/{target} pop {pop}")
                print(time.strftime("[%H:%M] ") + " | ".join(rows) + f" | {len(queue)} waiting")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nstopping runs; rerun the same command to resume")
        for p, *_ in running:
            p.terminate()


if __name__ == "__main__":
    main()
