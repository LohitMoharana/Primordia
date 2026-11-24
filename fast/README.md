# Primordia fast engine

The same world rules as stage 13 (seeing other agents, communication, seasons,
hazards, plus shared food on/off), about **110x faster** than the old engine.

Measured from the same saved state (old-engine checkpoint at tick 1,000, 1,588
agents), the same 100 ticks, median of repeated runs, same machine:

| engine | ticks/s | speed-up |
|---|---|---|
| old engine (1 core) | 6.1 | 1x |
| fast, NumPy only (no numba) | ~130 | ~22x |
| fast + numba, 1 core | ~418 | ~68x |
| fast + numba, 2 cores | ~674 | ~110x |

More cores help a single run further. When running many runs at once,
replicate_fast.py gives each run one core (68x each, all running in parallel).

## Setup (once)
Put this `fast` folder inside your `Primordia` folder (next to `lab/` and
`stages/`), then:

    pip install numpy numba

Without numba it still works, using a slower NumPy-only path (~22x).

## Run
    python fast/replicate_fast.py --seeds 2 3 4 5        # shared food ON and OFF, 4 seeds
    python fast/run_fast.py fast/runs/test --seed 1 --ticks 25000 --cofood   # one run

Options for run_fast.py: --cofood, --no-social, --no-comm, --no-seasons,
--no-hazards, --founders stage5|random. Runs resume from their last checkpoint.

## Survival test (same test as lab/evaluate.py)
    python fast/evaluate_fast.py fast/runs/cofood_on_seed2/checkpoint_25000.pkl fast/runs/cofood_off_seed2/checkpoint_25000.pkl

## What differs from the old engine
All agents sense the world at the start of a tick and act together, instead of
one after another. Where order mattered (several agents eating in one cell,
shared food, gifts) it is resolved explicitly in random order. So runs match
the old engine statistically, not bit for bit: from the same founders over
3,000 ticks, population, energy, generations and shared-food counts agree
(e.g. tick 1,500: 1,521 +/- 32 fast vs 1,520 old). The compiled (numba) path
gives exactly the same results as the NumPy path.

Checkpoints are saved every 2,500 ticks (old engine: every 500), so a run uses
about 80% less disk.

Files: engine.py (the simulator), kernels.py (compiled sensing+brain),
run_fast.py, replicate_fast.py, evaluate_fast.py, bench.py / eqcheck.py
(speed and equivalence checks, need the cloud's validation checkpoints).
