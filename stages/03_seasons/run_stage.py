"""
Stage 3: seasons on/off. Social sensing is on in both arms (adopted in
stage 1). Communication is on or off in BOTH arms according to the stage 2
verdict, passed in by the autopilot so the base stays consistent.

    python3 run_stage.py on  1 --communication 1
    python3 run_stage.py off 1 --communication 1
"""
import argparse
from sim import Simulation

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["on", "off"])
p.add_argument("seed", type=int)
p.add_argument("--communication", type=int, choices=[0, 1], required=True)
p.add_argument("--ticks", type=int, default=25000)
a = p.parse_args()

out_dir = f"runs/seasons_{a.arm}_seed{a.seed}"
sim = Simulation(seed=a.seed, out_dir=out_dir, social_sensing=True,
                 communication=bool(a.communication), seasons=(a.arm == "on"))
print(f"seasons={a.arm == 'on'} communication={bool(a.communication)} -> {out_dir}")
sim.run(max_ticks=a.ticks)
