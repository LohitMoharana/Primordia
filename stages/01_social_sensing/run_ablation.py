"""
The actual controlled experiment: TWO runs, same starting seed, same world
size, same population size, same brain architecture — the ONLY difference
is whether the brain's occupancy-sensing input carries real data or zeros.

Usage:
    python3 run_ablation.py on   seed=1   -> runs/ablation_on_seed1/
    python3 run_ablation.py off  seed=1   -> runs/ablation_off_seed1/

Run BOTH with the same seed number for a matched pair.
"""
import sys
from sim import Simulation

if __name__ == "__main__":
    condition = sys.argv[1]  # "on" or "off"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    assert condition in ("on", "off")

    social_sensing = (condition == "on")
    out_dir = f"runs/ablation_{condition}_seed{seed}"

    sim = Simulation(seed=seed, out_dir=out_dir, social_sensing=social_sensing)
    print(f"Running ablation condition={condition} (social_sensing={social_sensing}), "
          f"seed={seed}, writing to {out_dir}")
    sim.run(max_ticks=20000)
