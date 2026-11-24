import sys
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
from seed_from_stage5 import founders
from engine import FastSim, founders_from_old_agents
from modes import MODES
cfgs = {
    "A: 16 clusters, doors 8, life 600": dict(n_patches=16, patch_peak=0.057, door_width=8, patch_life=600.0),
    "B: 16 clusters, doors 8, life 400, speed 0.03": dict(n_patches=16, patch_peak=0.057, door_width=8, patch_life=400.0, patch_speed=0.03),
    "C: 4 big rooms (spacing 64), 8 clusters, doors 8": dict(wall_spacing=64, door_width=8),
}
for name, over in cfgs.items():
    res = []
    for s in (1, 2):
        kw = MODES["terrain_language"]["kw"](True); kw.update(over)
        sim = FastSim(seed=s, founders=founders_from_old_agents(founders(300, s, 48)), **kw)
        pops = []
        for k in range(8):
            sim.run(1000)
            pops.append(sim.population)
            if sim.population == 0: break
        res.append(pops)
    print(name, "| population every 1000 ticks:", res, flush=True)
