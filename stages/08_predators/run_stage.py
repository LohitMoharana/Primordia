"""
Stage 8: predators present/absent (ENVIRONMENT stage). Social sensing,
communication, seasons and hazards on in both arms (adopted); memory, kin
perception and sexual reproduction off (not adopted). Founders: the SAME 300
evolved stage-5 agents in both arms (seed_from_stage5.py), with the predator
input wired in at zero. Predators are scripted and do not evolve.

    python3 run_stage.py on  1
    python3 run_stage.py off 1
"""
import argparse
from sim import Simulation
from seed_from_stage5 import founders

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["on", "off"])
p.add_argument("seed", type=int)
p.add_argument("--ticks", type=int, default=25000)
a = p.parse_args()

out_dir = f"runs/predators_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=(a.arm == "on"))
print(f"predators={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
