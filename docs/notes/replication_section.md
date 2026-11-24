

## Replication across seeds (24 Nov 2026)

Kept stages (01, 02, 03, 05): seeds 2-5 run on the laptop with the original
engine (lab/replicate.py), evaluated with lab/evaluate.py (50 agents x 150
steps, closest-generation ON/OFF pair). Stage 13 (shared food): seeds 1-10 run
in the cloud with the fast engine (fast/, validated against the old engine,
see below), evaluated with fast/evaluate_fast.py at ticks 20,000 and 25,000.
Numbers are ON minus OFF; survival is steps survived out of 150.

### Social sensing (stage 1)
| seed | survival | eat-fraction |
|---|---|---|
| 1 | +5.7 to +8.8 (4 checks) | +0.33 to +0.35 |
| 2 | -2.2 | +0.40 |
| 3 | +0.2 | +0.09 |
| 4 | +4.7 | +0.04 |
| 5 | -8.5 | +0.29 |
Behaviour replicates: agents that can see others eat more on all 5 seeds.
Survival does not: positive on 2 of 4 new seeds, mean of new seeds -1.5.

### Communication (stage 2)
| seed | survival | eat-fraction |
|---|---|---|
| 1 | +6.1, +6.8 | n/a |
| 2 | -3.7 | +0.19 |
| 3 | +8.1 | +0.18 |
| 4 | -6.7 | -0.05 |
| 5 | +2.3 | +0.03 |
Mixed: survival positive on 2 of 4 new seeds (mean +0.0). Not a reliable survival effect.

### Seasons (stage 3), compared at the same tick (same point in the season cycle)
| seed | survival, tick 20,000 | survival, tick 25,000 | reproduce share ON / OFF |
|---|---|---|---|
| 2 | +11.9 | +26.3 | 7.7% / 3.5% |
| 3 | +11.4 | +4.8 | 6.4% / 2.8% |
| 4 | +16.2 | +9.9 | 4.3% / 5.4% |
| 5 | +6.7 | +21.0 | 8.8% / 7.3% |
Agents from the seasonal world survived longer at both readings on all 4 new
seeds (8 of 8). Caveat: each arm is tested in its own world, so part of the gap
may be the test conditions, not the agents. Seed 1's "seasonal agents move
twice as much" did not replicate (they moved less on 3 of 4 new seeds);
"reproduce more" mostly did (3 of 4). (The generation-matched readings, +29.9 /
-40.5 / +0.3 / +14.0, compared different points in the cycle and are not used.)

### Hazards (stage 5)
| seed | outcome | survival | time on poison vs chance |
|---|---|---|---|
| 1 | survived | (environment) | 0.51-0.69 |
| 2 | survived | -11.7 | 0.50 |
| 3 | ON arm extinct at tick 2,482 | n/a | n/a |
| 4 | ON arm extinct at tick 1,390 | n/a | n/a |
| 5 | survived | -9.8 | 0.24 |
Poison avoidance replicates on every seed that survived (3 of 3). But starting
a hazardous world from random brains failed on 2 of 4 new seeds, so the stage-5
founders used by stages 6-13 came from a lucky lineage. Living with poison
costs survival (expected for a harsher environment).

### Shared food (stage 13), fast engine, seeds 1-10
| seed | survival, 20k | survival, 25k | shared food eaten, ticks 0-5k -> 20-25k | wasted solo attempts per item per 1000 agents |
|---|---|---|---|---|
| 1 | +1.4 | +5.4 | 0.80 -> 0.86 | 8.78 -> 9.59 |
| 2 | -8.0 | +12.9 | 0.80 -> 0.88 | 8.57 -> 9.35 |
| 3 | +6.5 | -0.7 | 0.79 -> 0.91 | 8.84 -> 8.40 |
| 4 | +7.3 | +16.3 | 0.80 -> 0.88 | 8.96 -> 8.49 |
| 5 | +7.8 | -0.5 | 0.79 -> 0.92 | 9.04 -> 7.38 |
| 6 | +3.9 | -1.7 | 0.80 -> 0.88 | 8.92 -> 10.00 |
| 7 | -3.5 | -4.2 | 0.79 -> 0.90 | 8.84 -> 9.05 |
| 8 | +13.1 | -11.2 | 0.83 -> 0.92 | 8.34 -> 6.99 |
| 9 | -7.6 | -2.1 | 0.77 -> 0.90 | 9.32 -> 8.43 |
| 10 | -2.9 | -7.1 | 0.80 -> 0.85 | 8.84 -> 9.65 |
Survival: mean +1.8 (20k) and +0.7 (25k); ON better on 6/10 and 3/10 seeds.
The seed-1 old-engine survival advantage (+11.5, +5.2) does not replicate.
Eaten share rose on 10/10 seeds, but population also rises over the run, and on
seed 1 that rise tracked density (r = 0.90); wasted solo attempts per agent fell
on only 5/10. No evidence that agents learn to coordinate.

### Fast engine validation
From the same founders and settings, the fast engine matches the old engine:
over 3,000 ticks (5 fast seeds vs old seeds 1-2) population, energy,
generations and shared-food counts agree; over 15,000 ticks (old seed 1 vs fast
seed 1) generation 64 vs 68, mean age 363 vs 371, energy 2.87 vs 2.82. Late in
3,000-tick runs fast populations ran ~5-8% higher. Speed: old 6.1 ticks/s,
fast 418 (1 core, ~68x) and 674 (2 cores, ~110x) on the same machine and state.

### Overall
Behaviours replicate across seeds (eating more with social sensing, poison
avoidance, seasonal agents reproducing more). Most seed-1 survival advantages
do not: social sensing, communication and shared food are mixed across seeds.
Seasons is the one environment whose agents survived longer on every new seed.
Lesson: single-seed survival differences of 5-10 steps are within seed-to-seed
noise in this world; future verdicts need several seeds and behavioural measures.
