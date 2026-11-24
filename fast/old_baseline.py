import sys, time, csv
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
from sim import Simulation
from seed_from_stage5 import founders
arm, seed, ticks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = f"/home/claude/lab/fast/val/old_{arm}_seed{seed}"
sim = Simulation(seed=seed, out_dir=out, founders=founders(300, seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=False, markers=False,
                 learning=False, selfpred=False, language=False, cofood=(arm == "on"))
t0 = time.time()
sim.run(max_ticks=ticks)
print("ticks/s", ticks / (time.time() - t0))
