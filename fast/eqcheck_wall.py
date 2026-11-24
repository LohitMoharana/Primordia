import sys, copy, numpy as np
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
from seed_from_stage5 import founders
from engine import FastSim, founders_from_old_agents
from modes import MODES
F = founders_from_old_agents(founders(300, 1, 48))
base = FastSim(seed=1, founders=F, **MODES["terrain_language"]["kw"](True))
print("wall cells:", int(base.wall.sum()), "of", base.wall.size, "| agents on walls at start:", int(base.wall[base.x[base.alive], base.y[base.alive]].sum()))
for _ in range(50): base.step()
a = copy.deepcopy(base); b = copy.deepcopy(base); a.use_numba = False; b.use_numba = True
for t in range(200): a.step(); b.step()
print("numpy vs numba with walls, 200 ticks: pop", a.population, b.population, "| energies match:",
      a.population == b.population and np.allclose(a.energy[a.alive], b.energy[b.alive]),
      "| agents standing on walls:", int(b.wall[b.x[b.alive], b.y[b.alive]].sum()),
      "| food on walls:", float(b.food[b.wall > 0].sum()))
