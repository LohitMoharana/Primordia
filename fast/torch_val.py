import sys, time, csv, numpy as np
sys.path.insert(0, '/home/claude/lab/stages/13_cofood')
import torch
torch.set_num_threads(2)
from seed_from_stage5 import founders
from engine import founders_from_old_agents
from engine_torch import TorchSim
from modes import MODES
mode = sys.argv[1] if len(sys.argv) > 1 else "stage13"
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
for s in [1, 2, 3, 4, 5]:
    F = founders_from_old_agents(founders(300, s, 48)) if MODES[mode]["founders"] == "stage5" else None
    t0 = time.time()
    sim = TorchSim(seed=s, founders=F, out_dir=f"val/torch_{mode}_seed{s}", device="cpu", checkpoint_every=10**9,
                   **MODES[mode]["kw"](True))
    sim.run(ticks)
    r = sim.log_row()
    print(f"seed {s}: tick {r[0]} pop {r[1]} energy {r[2]:.2f} gen {r[4]} | {ticks/(time.time()-t0):.0f} ticks/s | {sim.cofood_stats}", flush=True)
