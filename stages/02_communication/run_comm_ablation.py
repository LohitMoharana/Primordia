"""
Controlled comm-channel experiment: same seed, same brain architecture
(communication=True vs False), only difference is whether a sent signal
ever actually reaches anyone. Agents can always CHOOSE to send — in the
"off" condition they just pay the cost for nothing, which is itself a
useful control (does a useless channel get pruned out by selection?).

Usage:
    python3 run_comm_ablation.py on  1
    python3 run_comm_ablation.py off 1
"""
import sys
from sim import Simulation

if __name__ == "__main__":
    condition = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    assert condition in ("on", "off")
    communication = (condition == "on")
    out_dir = f"runs/comm_{condition}_seed{seed}"
    sim = Simulation(seed=seed, out_dir=out_dir, social_sensing=True, communication=communication)
    print(f"Running communication={communication}, seed={seed}, -> {out_dir}")
    sim.run(max_ticks=20000)
