"""
Keeps the active stage's on/off runs going across environment restarts.

  python3 supervisor.py status   # show state of the active stage
  python3 supervisor.py run      # resume every unfinished run of the active
                                 # stage (bit-exact from its latest checkpoint)
                                 # up to target_tick, then mark it DONE

Run `run` with Bash run_in_background so it survives between replies.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys

LAB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(LAB)  # bundle root; code_dir in stages.json is relative to it


def _code(stage):
    cd = stage["code_dir"]
    return cd if os.path.isabs(cd) else os.path.join(ROOT, cd)
STAGES = os.path.join(LAB, "stages.json")
KEEP_EVERY = 2500  # prune checkpoints to this spacing (plus the latest two)


def load_stages():
    with open(STAGES) as f:
        return json.load(f)


def active_stage(data):
    for s in data["stages"]:
        if s["status"] == "running":
            return s
    return None


def checkpoints(run_dir):
    found = []
    for p in glob.glob(os.path.join(run_dir, "checkpoint_*.pkl")):
        m = re.search(r"checkpoint_(\d+)\.pkl$", p)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def last_log(run_dir):
    path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def prune(run_dir):
    ck = checkpoints(run_dir)
    keep = {t for t, _ in ck[-2:]} | {t for t, _ in ck if t % KEEP_EVERY == 0}
    for t, p in ck:
        if t not in keep:
            os.remove(p)


def is_alive(run_dir, run=None):
    """True if a python process is working on this run: either a resume
    (its command names the run dir) or a fresh launch (its command matches
    the run's launch arguments, which don't name the dir)."""
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    needles = [os.path.basename(run_dir.rstrip("/"))]
    if run and run.get("launch"):
        needles.append(" ".join(run["launch"][1:]))
    return any("python3" in line and "grep" not in line and any(n in line for n in needles)
               for line in out.splitlines())


def status():
    data = load_stages()
    s = active_stage(data)
    if s is None:
        print("No active stage.")
        return
    print(f"Active stage: {s['name']}  target_tick={s['target_tick']}  code={s['code_dir']}")
    for r in s["runs"]:
        d = os.path.join(_code(s), r["dir"])
        row = last_log(d)
        ck = checkpoints(d)
        done = os.path.exists(os.path.join(d, "DONE"))
        if os.path.exists(os.path.join(d, "EXTINCT")):
            done = "EXTINCT"
        print(f"  [{r['arm']:>3}] {r['dir']}: "
              f"tick={row['tick'] if row else '-'} gen={row['max_generation'] if row else '-'} "
              f"pop={row['population'] if row else '-'} "
              f"latest_ckpt={ck[-1][0] if ck else '-'} done={done} alive={is_alive(d, r)}")


def run():
    data = load_stages()
    s = active_stage(data)
    if s is None:
        print("No active stage.")
        return
    procs = []
    for r in s["runs"]:
        d = os.path.join(_code(s), r["dir"])
        if os.path.exists(os.path.join(d, "DONE")) or os.path.exists(os.path.join(d, "EXTINCT")):
            continue
        row = last_log(d)
        if row and int(row["population"]) == 0:
            open(os.path.join(d, "EXTINCT"), "w").close()
            print(f"[{r['arm']}] population is 0 at tick {row['tick']}: marked EXTINCT, not resuming")
            continue
        if is_alive(d, r):
            print(f"[{r['arm']}] already running, leaving it")
            continue
        ck = checkpoints(d)
        if not ck:
            cmd = r["launch"]
            print(f"[{r['arm']}] fresh launch: {' '.join(cmd)}")
        else:
            prune(d)
            tick, path = checkpoints(d)[-1]
            remaining = s["target_tick"] - tick
            if remaining <= 0:
                open(os.path.join(d, "DONE"), "w").close()
                print(f"[{r['arm']}] reached target, marked DONE")
                continue
            cmd = ["python3", "resume.py", path, r["dir"], "--ticks", str(remaining)]
            print(f"[{r['arm']}] resuming from tick {tick}, {remaining} ticks to go")
        os.makedirs(d, exist_ok=True)
        procs.append((r, d, subprocess.Popen(cmd, cwd=_code(s),
                                             stdout=subprocess.DEVNULL,
                                             stderr=open(os.path.join(d, "stderr.log"), "a"))))
    for r, d, p in procs:
        code = p.wait()
        ck = checkpoints(d)
        row = last_log(d)
        if row and int(row["population"]) == 0:
            open(os.path.join(d, "EXTINCT"), "w").close()
            print(f"[{r['arm']}] went EXTINCT at tick {row['tick']}")
            continue
        if code == 0 and ck and ck[-1][0] >= s["target_tick"]:
            open(os.path.join(d, "DONE"), "w").close()
            print(f"[{r['arm']}] finished, marked DONE")
        else:
            print(f"[{r['arm']}] exited with code {code}; see {d}/stderr.log")


if __name__ == "__main__":
    {"status": status, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
