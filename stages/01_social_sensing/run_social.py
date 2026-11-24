from sim import Simulation
import sys
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sim = Simulation(seed=seed, out_dir=f"runs/social{seed}")
sim.run(max_ticks=20000)
