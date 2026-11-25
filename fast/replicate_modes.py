"""
Run stage modes (modes.py) on many seeds, a few processes at a time, resumable.

    python3 replicate_modes.py --modes stage1 stage2 stage3 stage5 --seeds 1 2 3 4 5 6 7 8 9 10 --jobs 2

Output: fast/runs/<stage name>/<prefix>_<arm>_seed<N>/ (log.csv, checkpoints every 2500 ticks).
Runs that already reached their tick target, or died out, are skipped.
"""
import argparse, glob, os, re, subprocess, sys, time
from modes import MODES
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--modes", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--arms", nargs="+", default=["on", "off"])
ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
ap.add_argument("--engine", choices=["numba", "torch"], default="numba")
ap.add_argument("--ticks", type=int, default=None, help="override the mode's tick target")
ap.add_argument("--checkpoint-every", type=int, default=None)
ap.add_argument("--threads", type=int, default=os.cpu_count() or 2,
                help="threads shared by all running runs; as runs finish, the rest get more")
a = ap.parse_args()
# every run may use up to --threads; the share it actually uses is written to <run>/THREADS
# (idle worker threads sleep instead of spinning)
env = dict(os.environ, NUMBA_NUM_THREADS=str(max(1, a.threads)), OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")


def share_threads(running):
    """Split --threads over the running runs (at least 1 each) and tell each run its share."""
    if not running:
        return
    base, extra = divmod(a.threads, len(running))
    for i, item in enumerate(running):
        n = max(1, base + (1 if i < extra else 0))
        path = os.path.join(item[4], "THREADS")
        try:
            with open(path + ".tmp", "w") as f:
                f.write(str(n))
            os.replace(path + ".tmp", path)
        except OSError:
            pass


def finished(out, ticks):
    if os.path.exists(os.path.join(out, "EXTINCT")):
        return True
    ck = [int(re.search(r"checkpoint_(\d+)", f).group(1)) for f in glob.glob(os.path.join(out, "checkpoint_*.pkl"))]
    return bool(ck) and max(ck) >= ticks

queue = []
for s in a.seeds:
    for m in a.modes:
        for arm in a.arms:
            mo = MODES[m]
            out = os.path.join(HERE, "runs", mo["name"], f"{mo['prefix']}_{arm}_seed{s}")
            T = a.ticks or mo["ticks"]
            if not finished(out, T):
                queue.append((m, arm, s, out, T))
print(f"{len(queue)} runs to do, {a.jobs} at a time", flush=True)
running = []
t0 = time.time()
while queue or running:
    changed = False
    while queue and len(running) < a.jobs:
        m, arm, s, out, ticks = queue.pop(0)
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "run_fast.py"), out, "--mode", m, "--arm", arm,
               "--seed", str(s), "--ticks", str(ticks), "--engine", a.engine]
        if a.checkpoint_every:
            cmd += ["--checkpoint-every", str(a.checkpoint_every)]
        with open(os.path.join(out, "THREADS"), "w") as f:
            f.write("1")  # until share_threads() below gives the real share
        logh = open(os.path.join(out, "stdout.log"), "a")
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=logh, stderr=subprocess.STDOUT)
        logh.close()  # the child keeps its own handle
        running.append((p, m, arm, s, out, ticks))
        changed = True
    for item in list(running):
        p, m, arm, s, out, ticks = item
        if p.poll() is None:
            continue
        running.remove(item)
        changed = True
        logf = os.path.join(out, "log.csv")
        if p.returncode != 0 or not os.path.exists(logf):
            err = open(os.path.join(out, "stdout.log")).read().strip().splitlines()[-3:]
            print(f"[{time.time() - t0:5.0f}s] {m} {arm} seed{s}: FAILED (code {p.returncode}): " + " | ".join(err), flush=True)
            continue
        rows = open(logf).read().strip().splitlines()
        pop = int(float(rows[-1].split(",")[1])) if len(rows) > 1 else 0
        last = int(float(rows[-1].split(",")[0])) if len(rows) > 1 else 0
        if last < ticks - 10:
            pop = 0  # stopped before its target: died out
        if pop == 0:
            open(os.path.join(out, "EXTINCT"), "w").close()
        status = "EXTINCT" if pop == 0 else "done"
        print(f"[{time.time() - t0:5.0f}s] {m} {arm} seed{s}: {status}", flush=True)
    if changed:
        share_threads(running)
    time.sleep(1)
print("ALL DONE", flush=True)
