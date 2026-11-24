import sys
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
from seed_from_stage5 import founders
from engine import FastSim, founders_from_old_agents
for seed in [1, 2, 3, 4, 5]:
    F = founders_from_old_agents(founders(300, seed, 48))
    s = FastSim(seed=seed, founders=F, cofood=True, out_dir=f"val/fast_on_seed{seed}", checkpoint_every=10**9)
    s.run(3000)
    print(seed, s.log_row()[:3], s.cofood_stats, flush=True)
