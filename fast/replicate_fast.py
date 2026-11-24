"""
Run many fast-engine experiments at once (one process per run), on any OS.

    python fast/replicate_fast.py --seeds 6 7 8 9 10                 # shared food ON/OFF
    python fast/replicate_fast.py --seeds 2 3 --arms on --jobs 8

Each run: shared food ON (--cofood) or OFF, everything else as stage 13
(social sensing, communication, seasons, hazards on), founded by the same
stage-5 population. Output: fast/runs/cofood_<arm>_seed<N>/ (log.csv,
cofood.csv, a checkpoint every 2500 ticks). Safe to stop and rerun: runs
resume from their latest checkpoint and finished ones are skipped.
Each process uses one CPU thread (NUMBA_NUM_THREADS=1); --jobs sets how many
run at once (default: CPU threads minus 2).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--arms", nargs="+", choices=["on", "off"], default=["on", "off"])
ap.add_argument("--ticks", type=int, default=25000)
ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
a = ap.parse_args()

env = dict(os.environ, NUMBA_NUM_THREADS="1")
queue = [(arm, s) for s in a.seeds for arm in a.arms]
running = []
t0 = time.time()
while queue or running:
    while queue and len(running) < a.jobs:
        arm, s = queue.pop(0)
        out = os.path.join(HERE, "runs", f"cofood_{arm}_seed{s}")
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "run_fast.py"), out, "--seed", str(s),
               "--ticks", str(a.ticks)] + (["--cofood"] if arm == "on" else [])
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=open(os.path.join(out, "stdout.log"), "a"),
                             stderr=subprocess.STDOUT)
        running.append((p, arm, s))
        print(f"start {arm} seed{s}")
    for item in list(running):
        p, arm, s = item
        if p.poll() is not None:
            running.remove(item)
            print(f"{'done' if p.returncode == 0 else 'FAILED (see stdout.log)'} {arm} seed{s}"
                  f"  [{time.time() - t0:.0f} s]")
    time.sleep(2)
print("all finished")
