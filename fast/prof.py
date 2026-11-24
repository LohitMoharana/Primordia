import sys, time, cProfile, pstats
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
from seed_from_stage5 import founders
from engine import FastSim, founders_from_old_agents
F = founders_from_old_agents(founders(300, 1, 48))
s = FastSim(seed=1, founders=F, cofood=True)
for _ in range(800): s.step()
print('pop', s.population, 'cap', s.cap)
cProfile.run('for _ in range(200): s.step()', '/tmp/p')
pstats.Stats('/tmp/p').sort_stats('tottime').print_stats(12)
