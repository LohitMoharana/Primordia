"""
Stage 4: working memory on/off. Social sensing and communication on in both
arms (adopted). Seasons on or off in BOTH arms per the stage 3 verdict,
passed in by the autopilot.

    python3 run_stage.py on  1 --seasons 1
    python3 run_stage.py off 1 --seasons 1
"""
import argparse
from sim import Simulation

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["on", "off"])
p.add_argument("seed", type=int)
p.add_argument("--seasons", type=int, choices=[0, 1], required=True)
p.add_argument("--ticks", type=int, default=25000)
a = p.parse_args()

out_dir = f"runs/memory_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, social_sensing=True, communication=True,
                 seasons=bool(a.seasons), memory=(a.arm == "on"))
print(f"memory={a.arm == 'on'} seasons={bool(a.seasons)} -> {out_dir}")
sim.run(max_ticks=a.ticks)
