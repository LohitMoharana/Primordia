import sys, time, os, shutil
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
import numpy as np
CK = '/home/claude/lab/fast/val/old_on_seed1/checkpoint_1000.pkl'
T = int(sys.argv[1]) if len(sys.argv) > 1 else 100
which = sys.argv[2] if len(sys.argv) > 2 else 'both'
if which in ('both', 'old'):
    from sim import Simulation
    shutil.rmtree('/tmp/bench_old', ignore_errors=True); os.makedirs('/tmp/bench_old')
    s = Simulation.from_checkpoint(CK, out_dir='/tmp/bench_old')
    t0 = time.time()
    for _ in range(T): s.step()
    old = T / (time.time() - t0); print(f"old : {old:6.2f} ticks/s  pop {len(s.agents)}")
if which in ('both', 'fast'):
    import engine
    f = engine.from_old_checkpoint(CK, '/home/claude/lab/stages/13_cofood')
    for _ in range(5): f.step()   # warm-up
    t0 = time.time()
    for _ in range(T): f.step()
    fast = T / (time.time() - t0); print(f"fast: {fast:6.2f} ticks/s  pop {f.population}")
