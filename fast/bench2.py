"""Robust speed comparison: same saved state (old-engine checkpoint at tick 1000,
1,588 agents), same 100 ticks, repeated; reports the median ticks/s."""
import sys, time, os, shutil, copy, statistics, io, contextlib
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
CK = '/home/claude/lab/fast/val/old_on_seed1/checkpoint_1000.pkl'
T = 100
which = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
res = []
if which == 'old':
    from sim import Simulation
    for _ in range(reps):
        shutil.rmtree('/tmp/b_old', ignore_errors=True); os.makedirs('/tmp/b_old')
        with contextlib.redirect_stdout(io.StringIO()):
            s = Simulation.from_checkpoint(CK, out_dir='/tmp/b_old')
        t0 = time.perf_counter()
        for _ in range(T): s.step()
        res.append(T / (time.perf_counter() - t0))
else:
    import engine
    base = engine.from_old_checkpoint(CK, '/home/claude/lab/stages/13_cofood')
    w = copy.deepcopy(base)
    for _ in range(3): w.step()  # compile / warm-up
    for _ in range(reps):
        s = copy.deepcopy(base)
        t0 = time.perf_counter()
        for _ in range(T):
            s.step()
            if s.tick % 10 == 0: s.log_row()
        res.append(T / (time.perf_counter() - t0))
print(f"{which}: median {statistics.median(res):.1f} ticks/s  (runs: {', '.join(f'{r:.0f}' for r in res)})")
