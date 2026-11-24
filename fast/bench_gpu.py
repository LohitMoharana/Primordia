"""
Speed test on your laptop: numba engine (CPU, all threads) vs torch engine (GPU),
in a 128 x 128 world (stage-13 rules, no shared food) that grows to ~10,000 agents.

    python fast/bench_gpu.py

Writes the results to fast/bench_gpu_results.txt as well as printing them.
Takes a few minutes. Needs: pip install numba   and a CUDA build of torch, e.g.
    pip install torch --index-url https://download.pytorch.org/whl/cu128
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stages", "13_cofood"))

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
    with open(os.path.join(HERE, "bench_gpu_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

from seed_from_stage5 import founders
from engine import FastSim, founders_from_old_agents
import numba

KW = dict(size=128, social_sensing=True, communication=True, seasons=False, hazards=False,
          cofood=False, cofood_spawn_prob=0.0)
WARM, TIMED = int(os.environ.get("WARM", 1500)), int(os.environ.get("TIMED", 300))
F = founders_from_old_agents(founders(300, 1, 48))
say(f"world 128x128, warm-up {WARM} ticks, then time {TIMED} ticks")

def bench(name, sim):
    t0 = time.time()
    for _ in range(WARM):
        sim.step()
    warm = time.time() - t0
    p0 = sim.population
    t0 = time.time()
    for _ in range(TIMED):
        sim.step()
    if hasattr(sim, "device") and sim.device.type == "cuda":
        import torch; torch.cuda.synchronize()
    dt = time.time() - t0
    say(f"{name:<32} {TIMED / dt:7.1f} ticks/s at {p0}-{sim.population} agents (warm-up took {warm:.0f} s)")

bench(f"numba CPU ({numba.config.NUMBA_NUM_THREADS} threads)", FastSim(seed=1, founders=F, **KW))
try:
    import torch
    from engine_torch import TorchSim
    if torch.cuda.is_available():
        say(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
        bench("torch GPU", TorchSim(seed=1, founders=F, device="cuda", **KW))
    else:
        say(f"torch {torch.__version__} installed but no CUDA GPU visible: install a CUDA build "
            f"(pip install torch --index-url https://download.pytorch.org/whl/cu128)")
except ImportError:
    say("torch not installed: pip install torch --index-url https://download.pytorch.org/whl/cu128")
say("done")
