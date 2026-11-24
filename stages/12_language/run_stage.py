"""
Stage 12: richer language, two-symbol messages on/off (CAPABILITY stage).
Social sensing, communication, seasons and hazards on in both arms (adopted);
everything rejected so far off, including self-prediction (stage 11).
Founders: the SAME 300 evolved stage-5 agents in both arms; each gets a new
near-uniform second-symbol head and 25 zero input rows for the second-symbol
patch. When an agent signals it sends symbol 1 (its signal_N action) plus a
second symbol from the new head, same cost as before. ON: receivers sense both
symbols. OFF: only the first symbol is delivered.

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

out_dir = f"runs/language_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=False, markers=False,
                 learning=False, selfpred=False, language=(a.arm == "on"))
print(f"language={a.arm == 'on'} -> {out_dir}")
sim.run(max_ticks=a.ticks)
