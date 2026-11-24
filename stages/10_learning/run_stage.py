"""
Stage 10: lifetime learning on/off (CAPABILITY stage). Social sensing,
communication, seasons and hazards on in both arms (adopted); memory, kin
perception, sexual reproduction, predators and markers off (not adopted).
Founders: the SAME 300 evolved stage-5 agents in both arms, each given a
learning-rate gene of 0.01 that mutates in offspring. In ON, agents adjust
their output weights during life by energy gained or lost; children inherit
only the genome. In OFF the gene exists but is never used.

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

out_dir = f"runs/learning_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=False, markers=False,
                 learning=(a.arm == "on"))
print(f"learning={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
