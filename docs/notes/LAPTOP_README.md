# Primordia: laptop handoff (updated 22 Nov 2026, roadmap complete)

Everything from the cloud runs, ready to continue on your laptop.
`FINDINGS.md` has every result so far; `AUTOPILOT.md` is the procedure and roadmap.

**Two zips:**
- `primordia_laptop.zip`: all code (stages 01-11), every run's logs, the
  tools, and the saved stage-5 population that later stages start from. This is all
  you need to run new seeds.
- `primordia_stage11_resume.zip` (optional): the latest stage-11 checkpoints,
  so you can finish the cloud's seed-1 run exactly where it stopped. Unzip it
  into the same folder; it only adds files under `stages/11_selfpred/runs/`.

## 1. Set up (once)

Install Python 3.10+ (python.org; on Windows tick "Add Python to PATH"), then:

```bash
pip install numpy matplotlib
```

Commands below use `python`. On Linux/macOS, or if `python` isn't found, use
`python3` (or `py` on Windows). Run everything from the unzipped folder
(the one containing `lab/` and `stages/`).

Keep the laptop plugged in, on a hard surface, with the power mode set to
Performance. Turn off sleep while runs go overnight.

## 2. Time it first (2 minutes)

```bash
python lab/replicate.py --stages 05_hazards --seeds 99 --arms on --jobs 1
```

Let it print one progress line (every 5 minutes), note the tick it reached,
then Ctrl+C and delete `stages/05_hazards/runs/hazards_on_seed99`. Ticks per
5 minutes tells you how long one 25,000-tick run takes on one core.

## 3. Run many at once

`lab/replicate.py` runs one process per experiment, several at once (default:
your CPU threads minus 2, so 14 on a 16-thread Ryzen AI). Each experiment is
one stage, one arm (feature ON or OFF), and one seed.

The most valuable run is replicating the 4 kept features on new seeds.
Every result so far is from seed 1 only:

```bash
python lab/replicate.py --preset kept --seeds 2 3 4 5
```

That's 4 stages x 4 seeds x 2 arms = 32 runs. Then the rejected features
(seeds 2 and 3 are enough to check that the rejections hold):

```bash
python lab/replicate.py --preset rejected --seeds 2 3
```

Finish the cloud's stage-11 seed-1 run (needs the resume zip), plus new seeds:

```bash
python lab/replicate.py --stages 11_selfpred --seeds 1 2 3
```

Or everything: `python lab/replicate.py --preset all --seeds 2 3 4 5`.

**Stopping is safe.** Ctrl+C (or a reboot) stops it; run the same command again
and finished runs are skipped while interrupted ones resume exactly from their
last checkpoint (saved every 500 ticks). Runs that die out are marked EXTINCT
and skipped. Stages 01-05 start from random brains in a harsh world, so the odd
extinct seed is normal there. Stages 06-11 start from the saved stage-5
population (`stages/05_hazards/runs/hazards_on_seed1/checkpoint_23500.pkl`;
don't delete it).

Seed 1 for stages 01-10 is already done in the cloud (logs only, checkpoints
left out for size), so the runner skips seed 1 there. Use seeds 2+.

**Disk:** each checkpoint is 15-60 MB and runs keep every one. A finished run
can use 1-2 GB. To trim a finished run, keep only `checkpoint_*000.pkl` at
multiples of 2500 and the last one.

## 4. Read the results

```bash
python lab/summarize.py                       # every stage/seed with both arms
python lab/summarize.py --stages 05_hazards   # one stage
```

This prints ON-minus-OFF survival per seed and writes `lab/summary.csv`. A
feature holds up if most seeds point the same way as seed 1 (seed-1 numbers
are in FINDINGS.md: social sensing, communication, seasons, and hazards were
kept). One stage and seed in detail:

```bash
python lab/evaluate.py 05_hazards 3
python lab/hazard_avoidance.py stages/05_hazards runs/hazards_on_seed3/checkpoint_25000.pkl 100 150
```

Plots: `cd stages/05_hazards` then `python plot.py runs/hazards_on_seed3/log.csv`.
Clean duplicate log rows after resumes first: `python clean_log.py runs/hazards_on_seed3/log.csv`.

## 5. Where things stand

- Kept (seed 1): social sensing, communication, seasons, hazards.
- Not kept: memory, kin sharing (x2), sexual reproduction, predators, markers,
  lifetime learning.
- Stage 11 (self-prediction): not kept. Agents learn to predict themselves
  (error ~0.21 vs ~0.30) but survive worse in both readings and run at about half
  the population.
- Stage 12 (two-symbol messages): not kept. The second symbol never carried
  information about the sender's situation (six readings, all at noise level).
- Stage 13 (shared food that needs two eaters at once): not kept as a
  cooperation feature, since co-eating only rose with crowding. But the
  shared-food world survived better in both readings (+11.5 and +5.2 steps
  out of 150). Replicate it first: `python lab/replicate.py --preset promising --seeds 2 3 4 5`.
- The roadmap is complete (stages 01-13). Recommended next phase: speed up the
  simulator (batched NumPy or GPU), then a larger "sparse world" with clustered,
  drifting food, to re-test memory, language and shared food where information matters.

## 6. Layout and rules

```
FINDINGS.md   AUTOPILOT.md   LAPTOP_README.md
lab/          replicate.py (runner)  summarize.py (cross-seed table)  evaluate.py
              supervisor.py (cloud tool, Linux only)  *_analysis.py  stages.json
stages/NN_name/   each stage's complete, self-contained simulator + runs/
```

- Never edit a stage folder while it has runs going; copy it to a new folder.
- Checkpoints are Python pickles: only load ones you made yourself.
- Nothing here touches the network or anything outside this folder.
