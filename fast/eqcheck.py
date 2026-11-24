# same state + same random stream: NumPy path and Numba path must pick the same actions
import sys, copy, numpy as np
sys.path.insert(0,'/home/claude/lab/stages/13_cofood')
import engine
a = engine.from_old_checkpoint('/home/claude/lab/fast/val/old_on_seed1/checkpoint_1000.pkl','/home/claude/lab/stages/13_cofood')
b = copy.deepcopy(a); a.use_numba = False; b.use_numba = True
for t in range(200):
    a.step(); b.step()
same = (a.population == b.population) and np.allclose(a.energy[a.alive], b.energy[b.alive])
print("after 200 ticks: pop", a.population, b.population, "| energies match:", same)
