"""
Run one fast-engine experiment (resumable).

    python3 run_fast.py <out_dir> --seed 1 --ticks 25000 [--cofood] [--no-social]
        [--no-comm] [--no-seasons] [--no-hazards] [--founders stage5|random]

Founders: 'stage5' = the same 300 evolved stage-5 agents used by stages 6-13
(needs stages/13_cofood and the stage-5 checkpoint); 'random' = 80 random brains.
If <out_dir> already has checkpoints, the run resumes from the latest one.
"""
import argparse, glob, os, re, sys, time
import numpy as np
from engine import FastSim, founders_from_old_agents

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("out_dir")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--ticks", type=int, default=25000)
p.add_argument("--cofood", action="store_true")
p.add_argument("--no-social", action="store_true")
p.add_argument("--no-comm", action="store_true")
p.add_argument("--no-seasons", action="store_true")
p.add_argument("--no-hazards", action="store_true")
p.add_argument("--founders", choices=["stage5", "random"], default="stage5")
p.add_argument("--mode", default=None, help="stage1|stage2|stage3|stage5|stage13 (see modes.py); with --arm")
p.add_argument("--arm", choices=["on", "off"], default="on")
p.add_argument("--engine", choices=["numba", "torch"], default="numba", help="numba = CPU, torch = GPU if available")
p.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
p.add_argument("--checkpoint-every", type=int, default=None,
               help="ticks between saves (default 2500; 10000 for runs of 100k ticks or more)")
p.add_argument("--stage13-dir", default=os.path.join(HERE, "..", "stages", "13_cofood"))
a = p.parse_args()

# one process per run folder: a second copy of the same run exits at once (code 3)
import atexit
os.makedirs(a.out_dir, exist_ok=True)
_lock = os.path.join(a.out_dir, "RUNNING.lock")
try:
    os.close(os.open(_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
except FileExistsError:
    print(f"{a.out_dir} is already being run by another process (delete RUNNING.lock if that is wrong)")
    sys.exit(3)
atexit.register(lambda: os.path.exists(_lock) and os.remove(_lock))

ck = sorted(((int(re.search(r"checkpoint_(\d+)", f).group(1)), f)
             for f in glob.glob(os.path.join(a.out_dir, "checkpoint_*.pkl"))))
# skip unreadable saves (e.g. cut off by a kill before saves were atomic): use the newest good one
import pickle
while ck:
    try:
        with open(ck[-1][1], "rb") as _f:
            pickle.load(_f)
        break
    except Exception:
        bad = ck.pop()[1]
        os.replace(bad, bad + ".bad")
        print(f"unreadable save set aside: {bad}")
if ck:
    tick, path = ck[-1]
    if tick >= a.ticks:
        print("already done"); sys.exit(0)
    # drop log lines written after this checkpoint (the run was stopped before its next save),
    # so resuming does not duplicate ticks in log.csv / novelty.csv / births.csv / cofood.csv
    for name, col in (("log.csv", 0), ("novelty.csv", 0), ("births.csv", 0), ("cofood.csv", 0), ("predators.csv", 0)):
        f = os.path.join(a.out_dir, name)
        if os.path.exists(f):
            lines = open(f).read().splitlines()
            keep = [lines[0]] + [ln for ln in lines[1:] if ln and float(ln.split(",")[col]) <= tick]
            open(f, "w").write("\n".join(keep) + "\n")
    with open(path, "rb") as _f:
        _has_pred = "pred" in pickle.load(_f)
    if _has_pred:
        from predators import PredSim
        sim = PredSim.from_checkpoint(path, out_dir=a.out_dir)
    else:
        sim = FastSim.from_checkpoint(path, out_dir=a.out_dir)
    print(f"resuming from tick {tick}")
else:
    if a.mode:
        from modes import MODES
        m = MODES[a.mode]
        kw = m["kw"](a.arm == "on")
        founders_kind = m["founders"]
    else:
        kw = dict(social_sensing=not a.no_social, communication=not a.no_comm,
                  seasons=not a.no_seasons, hazards=not a.no_hazards, cofood=a.cofood)
        founders_kind = a.founders
    F = None
    if founders_kind == "stage5":
        sys.path.insert(0, os.path.abspath(a.stage13_dir))
        from seed_from_stage5 import founders
        F = founders_from_old_agents(founders(300, a.seed, 48))
    if a.engine == "torch":
        from engine_torch import TorchSim
        sim = TorchSim(seed=a.seed, out_dir=a.out_dir, founders=F, device=a.device, **kw)
    elif "predators" in kw:
        from predators import PredSim
        sim = PredSim(seed=a.seed, out_dir=a.out_dir, founders=F, **kw)
    else:
        sim = FastSim(seed=a.seed, out_dir=a.out_dir, founders=F, **kw)
sim.checkpoint_every = a.checkpoint_every or (10000 if a.ticks >= 100000 else 2500)
t0 = time.time()
sim.run(a.ticks - sim.tick, verbose=True)
print(f"done: tick {sim.tick}, population {sim.population}, {time.time() - t0:.0f} s")
