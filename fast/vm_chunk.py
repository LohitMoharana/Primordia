"""
Advance unfinished runs of the given modes for a limited wall time, then stop
(for environments that cannot keep background jobs). Each run saves every
5,000 ticks and resumes from its latest save next time.

    python3 vm_chunk.py --modes sparse2_memory0 --seeds 1 2 3 4 5 --jobs 2 --seconds 160
"""
import argparse, glob, os, re, subprocess, sys, time
from modes import MODES
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--modes", nargs="+", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--seconds", type=float, default=160)
a = ap.parse_args()

def progress(out):
    ck = [int(re.search(r"checkpoint_(\d+)", f).group(1)) for f in glob.glob(os.path.join(out, "checkpoint_*.pkl"))]
    return max(ck) if ck else 0

todo = []
for m in a.modes:
    mo = MODES[m]
    for s in a.seeds:
        for arm in ("on", "off"):
            out = os.path.join(HERE, "runs", mo["name"], f"{mo['prefix']}_{arm}_seed{s}")
            if os.path.exists(os.path.join(out, "EXTINCT")):
                continue
            pr = progress(out)
            log = os.path.join(out, "log.csv")
            if pr >= mo["ticks"]:
                continue
            if pr > 0 and os.path.exists(log):  # a run that died out keeps its last log row near 0
                last = open(log).read().strip().splitlines()[-1].split(",")
                if last[0] != "tick" and int(float(last[1])) == 0:
                    open(os.path.join(out, "EXTINCT"), "w").close(); continue
            todo.append((-pr, m, arm, s, out, mo["ticks"]))
todo.sort()  # most advanced first, so finished pairs appear sooner
env = dict(os.environ, NUMBA_NUM_THREADS="1")
procs = []
for _, m, arm, s, out, T in todo[:a.jobs]:
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "run_fast.py"), out, "--mode", m, "--arm", arm, "--seed", str(s),
           "--ticks", str(T), "--checkpoint-every", "5000"]
    procs.append((subprocess.Popen(cmd, cwd=HERE, env=env, stdout=open(os.path.join(out, "stdout.log"), "a"),
                                   stderr=subprocess.STDOUT), m, arm, s, out))
t0 = time.time()
while time.time() - t0 < a.seconds and any(p.poll() is None for p, *_ in procs):
    time.sleep(2)
for p, m, arm, s, out in procs:
    if p.poll() is None:
        p.terminate()
    log = os.path.join(out, "log.csv")
    last = open(log).read().strip().splitlines()[-1].split(",")[:2] if os.path.exists(log) else ["?", "?"]
    print(f"{m} {arm} seed{s}: saved up to tick {progress(out)} (log at {last[0]}, pop {last[1]})")
left = len(todo) - sum(1 for p, *_ in procs if p.returncode == 0)
print(f"runs still unfinished: {left}")
