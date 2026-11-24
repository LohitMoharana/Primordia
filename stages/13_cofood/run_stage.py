"""
Stage 13: shared food on/off (ENVIRONMENT stage). Social sensing,
communication, seasons and hazards on in both arms (adopted); everything
rejected so far off. LANGUAGE (stage 12's two-symbol messages) is set below
from stage 12's verdict.
Founders: the SAME 300 evolved stage-5 agents in both arms (25 zero input rows
for a new shared-food patch). Each tick, with probability 0.5, 4.0 energy enters
the world. ON: as a shared-food item that only two or more agents eating in
the same cell on the same tick can eat (they split it). OFF: as ordinary food
in 4 random cells. Same average energy in, only cooperation differs.

    python3 run_stage.py on  1
    python3 run_stage.py off 1
"""
import argparse
from sim import Simulation
from seed_from_stage5 import founders

LANGUAGE = False  # stage 12 verdict (two-symbol messages); set True only if adopted

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["on", "off"])
p.add_argument("seed", type=int)
p.add_argument("--ticks", type=int, default=25000)
a = p.parse_args()

out_dir = f"runs/cofood_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, founders=founders(300, a.seed, 48),
                 social_sensing=True, communication=True, seasons=True, memory=False,
                 hazards=True, kin=False, sexual=False, predators=False, markers=False,
                 learning=False, selfpred=False, language=LANGUAGE, cofood=(a.arm == "on"))
print(f"cofood={a.arm == 'on'} language={LANGUAGE} -> {out_dir}")
sim.run(max_ticks=a.ticks)
