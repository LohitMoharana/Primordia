"""
Stage 6b: kin perception on/off with PRODUCTIVE sharing (receiver gets 2x
what the giver gives; stage 6 had benefit <= cost so altruism could not evolve).
Capability stage. Social sensing,
communication, seasons and hazards on in both arms (adopted); memory off.
Founders: the SAME 300 evolved agents from stage 5 in both arms (see
seed_from_stage5.py). Random founders went extinct three times here.
Both arms have heritable family tags and the share action; only whether
agents can SEE relatedness differs (the kin patch is zeroed when off).

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

out_dir = f"runs/kin_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True,
                 seasons=True, memory=False, hazards=True, kin=(a.arm == "on"))
print(f"kin={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
