"""
Primordia lab queue: runs queued experiments one after another, for as long as it is left running.

    python fast/lab_queue.py          (leave it running; Ctrl+C to stop)

Jobs are small JSON files in fast/queue/pending/, run in name order:

    {"script": "modes_compare.py", "args": ["--worlds", "start", "--seeds", "10"], "note": "why"}

Only Primordia's own experiment scripts in fast/ can be run (SCRIPTS below), with plain
arguments and no shell, so a job file cannot run anything else. While a job runs it sits in
running/; afterwards it moves to done/ or failed/, and its full output is in logs/<job>.log.
heartbeat.txt is rewritten every minute (so it is easy to see that the queue is alive).
A job interrupted by a stop is started again next time (all experiment scripts resume).
To stop after the current job: create fast/queue/STOP.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "queue")
DIRS = {k: os.path.join(Q, k) for k in ("pending", "running", "done", "failed", "logs")}
SCRIPTS = {"modes_compare.py", "modes_search.py", "asal_search.py", "asal_illuminate.py",
           "replicate_modes.py", "run_fast.py", "analyze_memseq.py"}


def beat(msg):
    with open(os.path.join(Q, "heartbeat.txt"), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def load(path):
    with open(path) as f:
        job = json.load(f)
    script, args = job.get("script"), job.get("args", [])
    if script not in SCRIPTS:
        raise ValueError(f"script {script!r} is not one of the allowed experiment scripts")
    if not isinstance(args, list) or not all(isinstance(x, (str, int, float)) for x in args):
        raise ValueError("args must be a list of plain values")
    return script, [str(x) for x in args]


def run(name):
    src = os.path.join(DIRS["running"], name)
    log = os.path.join(DIRS["logs"], name.replace(".json", ".log"))
    try:
        script, args = load(src)
    except Exception as e:
        with open(log, "a") as f:
            f.write(f"rejected: {e}\n")
        os.replace(src, os.path.join(DIRS["failed"], name))
        return
    cmd = [sys.executable, os.path.join(HERE, script)] + args
    beat(f"running {name}: {' '.join(cmd[1:])}")
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=os.path.dirname(HERE))
        while p.poll() is None:
            time.sleep(60)
            beat(f"running {name} for {(time.time() - t0) / 60:.0f} min")
        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} exit code {p.returncode} after {(time.time() - t0) / 60:.0f} min\n")
    os.replace(src, os.path.join(DIRS["done" if p.returncode == 0 else "failed"], name))


def main():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    for name in os.listdir(DIRS["running"]):  # interrupted last time: run again (scripts resume)
        os.replace(os.path.join(DIRS["running"], name), os.path.join(DIRS["pending"], name))
    print("lab queue running; waiting for jobs in", DIRS["pending"], flush=True)
    while True:
        if os.path.exists(os.path.join(Q, "STOP")):
            os.remove(os.path.join(Q, "STOP"))
            beat("stopped (STOP file)")
            print("stopped", flush=True)
            return
        jobs = sorted(n for n in os.listdir(DIRS["pending"]) if n.endswith(".json"))
        if not jobs:
            beat("idle, waiting for jobs")
            time.sleep(60)
            continue
        name = jobs[0]
        os.replace(os.path.join(DIRS["pending"], name), os.path.join(DIRS["running"], name))
        print(time.strftime("%H:%M"), "start", name, flush=True)
        run(name)
        print(time.strftime("%H:%M"), "finished", name, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        beat("stopped (Ctrl+C)")
