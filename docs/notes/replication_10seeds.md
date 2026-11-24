

## Replication to 10 seeds with the fast engine (24 Nov 2026)

The four kept stages were re-run on seeds 1-10 in the fast engine, using a
mode per stage (fast/modes.py) with that stage's original settings: brain
size (stage 1: 16 hidden units), action set (7 or 11 actions), starting-weight
scale, 80 random founders, no shared food. Both arms were tested at the same
two ticks (stages 1-2: 17,500 and 20,000; stages 3 and 5: 22,500 and 25,000),
50 agents x 150 steps each. Numbers are ON minus OFF; survival is steps
survived out of 150.

Check against the old engine (seeds 1-5, same settings): population histories
match stage by stage. Same early crash, recovery at the same ticks, same final
sizes (e.g. social sensing ON at tick 20,000: fast 1,470-1,576 vs old
1,466-1,602). Old "seasons off" and "communication on" are the same settings
and match each other exactly in both engines. Hazards from random brains
passes through a bottleneck of a few agents near tick 1,000 in both engines;
extinction rate: old 2 of 5, fast 3 of 5 (on seeds 1-5).

### Social sensing (stage 1), fast engine
| seed | survival (2 readings) | eat-fraction | move share ON / OFF |
|---|---|---|---|
| 1 | -11.6, -12.0 | +0.16, +0.14 | 13.8% / 31.9% |
| 2 | -1.8, -5.8 | +0.07, +0.09 | 14.7% / 24.1% |
| 3 | +0.6, +5.1 | +0.34, +0.30 | 17.4% / 44.2% |
| 4 | -7.3, -4.1 | +0.26, +0.34 | 14.1% / 39.3% |
| 5 | -4.9, -1.8 | +0.18, +0.22 | 19.5% / 37.5% |
| 6 | -13.0, -0.5 | -0.10, -0.05 | 13.2% / 15.5% |
| 7 | +2.2, -15.6 | +0.28, +0.33 | 21.8% / 39.8% |
| 8 | -14.5, -19.3 | -0.03, +0.01 | 14.1% / 19.1% |
| 9 | -7.0, -2.0 | +0.28, +0.14 | 13.9% / 30.8% |
| 10 | +17.8, -0.1 | +0.25, +0.29 | 12.2% / 33.9% |
Mean survival -4.0 and -5.6. Agents that see others eat more (8/10 seeds) and
move much less (10/10). No survival advantage; if anything a small cost.

### Communication (stage 2), fast engine
| seed | survival | eat-fraction | move ON / OFF |
|---|---|---|---|
| 1 | -14.0, +1.7 | +0.02, +0.02 | 9.0% / 14.2% |
| 2 | -2.7, -7.1 | +0.02, +0.07 | 10.8% / 12.7% |
| 3 | +19.0, +4.5 | +0.10, +0.10 | 7.1% / 13.3% |
| 4 | -3.5, -16.7 | -0.02, +0.05 | 12.2% / 12.3% |
| 5 | +0.7, +4.1 | +0.14, +0.10 | 10.3% / 13.1% |
| 6 | +1.3, +1.7 | -0.02, -0.03 | 9.1% / 10.8% |
| 7 | +11.2, -6.6 | +0.01, -0.02 | 13.6% / 10.8% |
| 8 | +8.8, -8.7 | +0.05, -0.01 | 9.2% / 11.9% |
| 9 | -9.6, +4.0 | -0.03, -0.01 | 7.4% / 14.8% |
| 10 | +1.7, +17.4 | +0.11, +0.08 | 10.3% / 14.5% |
Mean survival +1.3 and -0.6. No reliable effect. Agents that can signal move
a little less (9/10 seeds).

### Seasons (stage 3), fast engine
| seed | survival | reproduce share ON / OFF (25k) | population ON / OFF (25k) |
|---|---|---|---|
| 1 | +26.4, +18.7 | 2.6% / 7.9% | 1914 / 1418 |
| 2 | +22.8, +18.0 | 7.4% / 1.7% | 1730 / 1428 |
| 3 | +14.0, +15.8 | 1.6% / 3.7% | 1681 / 1442 |
| 4 | +24.7, +31.5 | 6.5% / 2.1% | 1863 / 1399 |
| 5 | +18.2, +12.9 | 4.0% / 3.3% | 1785 / 1441 |
| 6 | +21.7, +10.9 | 4.8% / 5.8% | 1885 / 1453 |
| 7 | +27.5, +14.2 | 7.0% / 4.1% | 1769 / 1469 |
| 8 | +13.9, +8.2 | 1.7% / 4.0% | 1759 / 1416 |
| 9 | +8.9, +4.4 | 2.2% / 11.6% | 1778 / 1447 |
| 10 | +34.2, +11.5 | 3.8% / 6.2% | 1809 / 1440 |
Survival higher in the seasonal world at all 20 readings (mean +21.2 and
+14.6), the only consistent survival effect in the project (old engine seeds
2-5: 8 of 8). Caveat unchanged: each arm is tested in its own world, and both
readings land at the same two points of the 600-tick cycle on every seed
(tick 22,500: regrowth crossing from plentiful to lean; tick 25,000: lean
half), just after the seasonal population peak (ON population is above OFF
there). Part of the gap may be those test conditions rather than the agents;
readings at other points of the cycle would settle it. Reproduce share does not differ
consistently (old-engine "reproduce more" does not hold at 10 seeds).

### Hazards (stage 5), fast engine
| seed | outcome | survival | time on poison vs chance |
|---|---|---|---|
| 1, 2, 4, 8, 10 | ON arm extinct (founding from random brains) | n/a | n/a |
| 3 | survived | -0.6, -6.8 | 0.23, 0.19 |
| 5 | survived | +5.0, -5.2 | 0.30, 0.38 |
| 6 | survived | -1.9, +5.0 | 0.46, 0.38 |
| 7 | survived | -4.7, -10.4 | 0.48, 0.31 |
| 9 | survived | -10.9, -17.3 | 0.69, 0.53 |
Poison avoidance on every surviving seed (10 of 10 readings well below 1;
old engine 3 of 3). Founding a hazardous world from random brains fails half
the time (fast 5 of 10, old 2 of 4 new seeds). Surviving with poison costs some
survival (mean -2.6, -6.9), as expected for a harsher world.

### Verdicts after 10 seeds
| Feature | Holds up | Does not |
|---|---|---|
| Social sensing | Eat more, move much less | Survival gain (seed 1 only) |
| Communication | Move slightly less | Any survival effect |
| Seasons | Higher survival in all 20 readings (with the test-conditions caveat) | Seed 1's behaviour details (move more, reproduce more) |
| Hazards | Poison avoidance, every surviving seed | Founding reliability: half the worlds die out |


### Correction: what the survival test measures (24 Nov, same day)
The standard survival test (50 agents x 150 steps, each arm in its own world)
mixes three things: the brains, each population's age and energy at the
moment of the snapshot (agents die of old age at 500), and the state of that
world at that moment. Checks:
* Seasons tested at other points of the cycle, continuing each run from tick
  25,000: survival ON-OFF was -19.5 (tick 25,150), -32.4 (25,300) and +14.8
  (25,450), 0/10, 0/10 and 10/10 seeds. The sign follows the season, so the
  "+21.2 / +14.6 at all 20 readings" result above was an artifact of when the
  test ran. At tick 25,000 the seasonal populations are younger (mean age 205
  vs 247), so fewer of them die of old age during the test.
* Fair test (fast/cross_test2.py): both arms' agents in the SAME world, all
  starting with energy 5 and age 100. Then everyone survives the 150 steps,
  so the measure is energy gained and eat-fraction.
* Lesion test (fast/lesion.py): the ON arm's own agents in their own world,
  with the feature's information and with it removed. A drop means the
  brains really use it.

| Feature | Fair test, same world (energy ON-OFF; eat-fraction) | Lesion test (cost of removing the information) |
|---|---|---|
| Social sensing | in a world without social info: -1.38 (1/10); with it: +0.36 (5/10), eat +0.29 (9/10) | energy -0.59 (worse on 9/10), eat-fraction -0.32 (10/10) |
| Communication | without signals: +0.28 (5/10); with: +0.48 (7/10), eat +0.09 (9/10) | energy +0.31 (worse on 3/10), eat-fraction -0.01 (6/10): not used |
| Seasons | steady world: +0.09 (4/10), eat -0.04 (0/10 higher); seasonal world: +0.02 (7/10) | n/a (environment) |
| Shared food | -0.12 (5/10) and -0.09 (4/10) | n/a |
(In the "with" columns, brains from the OFF arm receive inputs they never
evolved to use, which can handicap them, so the lesion test is the cleaner one.)

### Verdicts after 10 seeds (replaces the table above)
| Feature | Verdict |
|---|---|
| Social sensing | Real and used: removing the information cuts eating on 10/10 seeds and energy on 9/10. Its survival gains in the old test were not reliable. |
| Communication | Not used: agents signal, but removing the signals changes nothing. |
| Seasons | No better brains: same energy gain in the same world; seasonal brains eat slightly less often. Survival differences came from age structure and test timing. |
| Hazards | Poison avoidance is real (every surviving seed); founding from random brains fails half the time. |
| Shared food | No effect on brains, no learned coordination. |
From now on, verdicts use the fair and lesion tests on several seeds.
