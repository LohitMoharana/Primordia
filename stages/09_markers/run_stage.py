"""
Stage 9: world markers / external memory on/off (CAPABILITY stage).
Social sensing, communication, seasons and hazards on in both arms (adopted);
memory, kin perception, sexual reproduction and predators off (not adopted).
Founders: the SAME 300 evolved stage-5 agents in both arms. "mark" exists and
costs energy in both arms; only in ON can marks be sensed.

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

out_dir = f"runs/markers_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=False, markers=(a.arm == "on"))
print(f"markers={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
