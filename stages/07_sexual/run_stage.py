"""
Stage 7: two-parent (sexual) reproduction on/off (capability stage).
Social sensing, communication, seasons and hazards on in both arms; memory
and kin perception off (not adopted). Founders: the SAME 300 evolved stage-5
agents in both arms (seed_from_stage5.py). The "mate" action exists in both
arms; it costs the initiator exactly what asexual "reproduce" costs and mixes
both parents' genomes; in OFF it just fails.

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

out_dir = f"runs/sexual_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=(a.arm == "on"))
print(f"sexual={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
