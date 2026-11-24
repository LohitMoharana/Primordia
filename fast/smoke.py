import sys, time
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
from seed_from_stage5 import founders
from engine import FastSim, founders_from_old_agents
F = founders_from_old_agents(founders(300, 1, 48))
for cof in [True, False]:
    s = FastSim(seed=1, founders=F, cofood=cof)
    t0 = time.time()
    for i in range(1, 2001):
        s.step()
        if i % 500 == 0:
            r = s.log_row()
            print(f"cofood={cof} tick {i} pop {r[1]} energy {r[2]:.2f} gen {r[4]} | {i/(time.time()-t0):.0f} ticks/s")
    print(s.cofood_stats)
