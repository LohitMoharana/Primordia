import sys, time, copy, numpy as np
sys.path.insert(0,'/home/claude/lab/stages/13_cofood')
import engine, kernels
U = kernels._make_sense_think_unrolled(engine.HIDDEN_SIZE, engine.OUTPUT_SIZE)
base = engine.from_old_checkpoint('/home/claude/lab/fast/val/old_on_seed1/checkpoint_1000.pkl','/home/claude/lab/stages/13_cofood')
for _ in range(5): base.step()
# exactness: same state, array kernel vs unrolled kernel, 200 ticks
a = copy.deepcopy(base); b = copy.deepcopy(base)
orig = engine.sense_think
for t in range(200):
    engine.sense_think = orig; a.step()
    engine.sense_think = U; b.step()
print("200 ticks: pop", a.population, b.population, "energies match:",
      a.population == b.population and np.allclose(a.energy[a.alive], b.energy[b.alive]))
for name, fn in [("array", orig), ("unrolled", U), ("array", orig), ("unrolled", U)]:
    engine.sense_think = fn
    s = copy.deepcopy(base)
    for _ in range(3): s.step()
    t0 = time.perf_counter()
    for _ in range(60): s.step()
    print(name, round(60 / (time.perf_counter() - t0)), "ticks/s, pop", s.population)
