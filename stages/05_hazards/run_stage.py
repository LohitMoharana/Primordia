"""
Stage 5: hazards present/absent (environment stage). Social sensing,
communication and seasons on in both arms (adopted); memory off (rejected
in stage 4). Both arms have the hazard-sensing input; with hazards off it
always reads zero, so the brains are identical.

    python3 run_stage.py on  1
    python3 run_stage.py off 1
"""
import argparse
from sim import Simulation

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["on", "off"])
p.add_argument("seed", type=int)
p.add_argument("--ticks", type=int, default=25000)
a = p.parse_args()

out_dir = f"runs/hazards_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, social_sensing=True, communication=True,
                 seasons=True, memory=False, hazards=(a.arm == "on"))
print(f"hazards={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
