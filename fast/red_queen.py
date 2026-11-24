"""
Red Queen test for the arms race: pit predators from one time point against
prey from another, in the same world, and count kills.

For one run and a pair (prey checkpoint A, predator checkpoint B):
  * the world and prey come from A; a fixed sample of N_PREY prey is kept
    (so prey density is equal in every pairing);
  * N_PRED predators are drawn from B and dropped at the same random spots;
  * predators cannot breed, starve or age out, and no newcomers arrive, so
    the count measures hunting skill against evasion skill only;
  * run TICKS ticks, report kills per predator per 100 ticks.

    python3 red_queen.py runs/40_arms_race/predevolve_on_seed1 50000 1000000

Reading the 2x2 table (early/late prey x early/late predators):
  late predators beat early predators on the same prey -> predators improved
  early predators catch fewer late prey than early prey -> prey improved
"""
import sys

import numpy as np

from predators import PredSim

N_PREY, N_PRED, TICKS = 400, 100, 300


def trial(prey_ckpt, pred_ckpt, rep=0):
    sim = PredSim.from_checkpoint(prey_ckpt)
    r = np.random.default_rng(1000 + rep)
    alive = np.flatnonzero(sim.alive)
    keep = r.choice(alive, min(N_PREY, len(alive)), replace=False)
    sim.alive[:] = False
    sim.alive[keep] = True
    other = PredSim.from_checkpoint(pred_ckpt).pred
    pick = r.integers(0, len(other["x"]), N_PRED)
    S = sim.size
    sim.pred = dict(x=r.integers(0, S, N_PRED), y=r.integers(0, S, N_PRED), e=np.full(N_PRED, 8.0),
                    age=np.zeros(N_PRED, np.int64), gen=np.zeros(N_PRED, np.int64),
                    **{k: other[k][pick].copy() for k in ("w1", "b1", "w2", "b2")})
    # no breeding, no newcomers; 300 ticks is well inside the 800-tick lifespan, so nobody ages out
    sim.cfg = dict(sim.cfg, pred_floor=0, pred_rep_min=1e9, pred_delay=0)
    sim.rng = np.random.default_rng(2000 + rep)
    k0 = sim.pred_stats["kills"]
    for _ in range(TICKS):
        sim.step()
        sim.pred["e"] = np.maximum(sim.pred["e"], 1.0)  # nobody starves during the test
    return (sim.pred_stats["kills"] - k0) / N_PRED / TICKS * 100


def table(run_dir, early, late, reps=2):
    ck = lambda t: f"{run_dir}/checkpoint_{t}.pkl"
    out = {}
    for py, pp in ((early, early), (early, late), (late, early), (late, late)):
        out[(py, pp)] = float(np.mean([trial(ck(py), ck(pp), rep) for rep in range(reps)]))
    return out


if __name__ == "__main__":
    d, e, l = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    t = table(d, e, l)
    for (py, pp), v in t.items():
        print(f"prey@{py} vs predators@{pp}: {v:.2f} kills per predator per 100 ticks")


def blind_trial(prey_ckpt, pred_ckpt, rep=0, blind=True):
    """Same as trial(), but prey cannot sense predators (lesion of inputs 125-149)."""
    orig = PredSim.step
    def step(self):
        orig(self)
    sim_cls_step = PredSim.step
    if blind:
        def step_blind(self):
            pred_on = self.cfg.get("predators", False)
            self.cfg["predators"] = False      # prey sense nothing...
            FastStep(self)
            self.cfg["predators"] = pred_on
            self._pred_step()                  # ...but predators still hunt
        PredSim.step = step_blind
    try:
        return trial(prey_ckpt, pred_ckpt, rep)
    finally:
        PredSim.step = sim_cls_step


from engine import FastSim as _F
def FastStep(self):
    self.pred_sense = None
    _F.step(self)
