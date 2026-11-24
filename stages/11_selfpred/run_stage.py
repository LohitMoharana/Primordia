"""
Stage 11: self-prediction on/off (CAPABILITY stage; a marker for
self-modelling, NOT a test of consciousness). Social sensing, communication,
seasons and hazards on in both arms (adopted); everything rejected so far off.
Founders: the SAME 300 evolved stage-5 agents in both arms, each given a small
random prediction head. Every tick each agent predicts its own next-tick
energy. In ON, prediction error costs energy (0.03 per unit of error); in OFF
the prediction is made but has no cost or effect, so its accuracy is the
neutral baseline.

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

out_dir = f"runs/selfpred_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=False, markers=False,
                 learning=False, selfpred=(a.arm == "on"))
print(f"selfpred={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
