# Findings log

All comparisons: same seed, same starting population, identical brain
architecture in both arms; only the stage's flag differs. 50 agents sampled
per arm, each run for 150 steps on the arm's own world snapshot. Single seed
(seed 1) per stage because of 2 CPU cores, so treat effects as suggestive
until replicated on more seeds.

## Stage 0: baseline foraging (no sensing of others)
- Random-brain agents evolved foraging with no hand-coded rules.
- Gen 1 vs gen 60 on the same food layout: gen 1 lost energy (4.22 -> 2.97);
  gen 60 gained (1.87 -> 4.45) and almost never idled.
- 76% of a random sample of 50 gen-60 agents gained energy over 150 steps.

## Stage 1: social sensing (see crowding in a 5x5 patch)
Arms: ablation_on_seed1 vs ablation_off_seed1 (code: /home/claude/alife).

| Check (gen) | Eat-fraction ON-OFF | Survival ON-OFF (of 150) | Energy ON-OFF |
|---|---|---|---|
| ~60 | +0.330 | +8.8 | -0.08 |
| ~74 | +0.349 | +8.1 | +0.19 |
| ~82 | +0.346 | +5.7 | -0.60 |
| ~88 | +0.352 | +8.5 | -0.37 |

Verdict: ADOPTED. Social sensing reliably changes behaviour (agents eat about
twice as often) and improves survival by about 6-9 steps in 150, consistently
across four checks. Net energy gain is not improved (noise around zero, mostly
slightly negative), which fits a contention story: sensing crowds pulls agents
to food others are also converging on, so they eat more often but less per bite.
That story is a hypothesis, not tested.

## Stage 2: communication channel (4 symbols, one-tick broadcast, costs energy)
Arms: comm_on_seed1 vs comm_off_seed1 (code: /home/claude/alife_comm). Social
sensing on in both. In OFF, agents can still choose signal actions and pay
for them, but nobody ever receives them.

| Check (gen) | Eat-fraction ON-OFF | Survival ON-OFF | Energy ON-OFF | Signal share ON / OFF |
|---|---|---|---|---|
| 41 (early, below verdict depth) | +0.070 | -4.5 | -1.11 | 6.8% / 6.5% |
| 69 / 67 | +0.100 | +6.1 | -0.23 | 3.7% / 3.0% |
| 79 / 78 | +0.121 | +6.8 | +0.76 | 3.6% / 2.6% |

Early read: no sign the channel is being used for anything. Signal use is
the same whether or not anyone can hear it, so it looks like leftover random
behaviour. ON is slightly worse so far, plausibly because meaningless signals
add noise to inputs. Gen 67-69: survival flipped to favour ON (+6.1), eat-fraction gap grew. Signal
use fell in both arms (6.8% -> 3.7% ON, 6.5% -> 3.0% OFF), consistent with
selection pruning a costly action; ON keeps slightly more. Mixed so far; one
more reading needed before a verdict.

Meaning check (gen 79, ON arm, 200 agents x 60 steps): which symbol an agent
sends depends on its situation. Average food under the sender: symbol 1 0.300,
symbol 2 0.231, symbol 3 0.245, symbol 4 0.150 (roughly 5 standard errors
between symbols 1 and 4). So signals carry information about the sender's
surroundings, mainly "plenty here" (1) vs "little here" (4). Not tested: whether
receivers act on this, i.e. whether the survival benefit comes from the
meaning or just from signals marking where active agents are.

Verdict: ADOPTED. Survival favoured ON at both verdict-depth checks (+6.1,
+6.8), energy turned positive at the deeper check, no population cost.
Signalling is costly and was pruned in both arms, but less where it can be
heard. Single seed.

## Stage 3: seasons (regrowth rate cycles over 600 ticks, same average in both arms)
Arms: seasons_on_seed1 vs seasons_off_seed1 (code: lab/stages/03_seasons).
Social sensing and communication on in both.

| Check (gen) | Eat-fraction ON-OFF | Survival ON-OFF | Energy ON-OFF | Population ON / OFF | Move share ON / OFF |
|---|---|---|---|---|---|
| 61 / 61 | -0.151 | -12.7 | +0.73 | 843 / 1427 | 14.0% / 6.5% |
| 91 / 91 | -0.120 | +9.5 | -0.62 | 973 / 1437 | 12.8% / 7.2% |

Seasons is an environment change, not a capability, so lower survival and a
smaller population in the ON arm are expected and are not a reason to reject it
(the rule in AUTOPILOT.md has been amended for environment stages). Behaviour
differs clearly: seasonal agents move about twice as much and eat less often,
yet gain more energy per test, suggesting they evolved to roam for patches
rather than graze in place. Caveat: tests run on a world snapshot at one point
in the season cycle, so energy numbers depend on the phase.

Second reading (gen 91): survival and energy flipped sign relative to gen 61,
which fits the phase caveat: the test window lands at a different point in
the season cycle each time, so those two numbers mostly reflect the season,
not the agents. The behavioural differences held at both depths: seasonal
agents move about twice as much (12.8-14.0% vs 6.5-7.2% of actions), eat less
often, and at gen 91 reproduce about three times as often (9.6% vs 3.2%),
consistent with breeding when food is plentiful. Population after tick 5000:
ON 519-1744 (mean 1093), swinging with the seasons; OFF 1205-1511 (mean 1384).
No collapse over 25,000 ticks.

Verdict: ADOPTED as an environment. Sustainable, and it produces distinct
behaviour (roaming, likely boom-time breeding). Seasons stays on in later
stages, which gives stage 4 (memory) something worth remembering.

## Stage 4: working memory (recurrent hidden state carried between ticks)
Arms: memory_on_seed1 vs memory_off_seed1 (code: lab/stages/04_memory).
Social sensing, communication and seasons on in both. In OFF the brain has the
same recurrent weights (wh) but its state is wiped every tick, so wh does
nothing and only drifts by mutation: OFF's |wh| is the neutral-drift baseline.

| Check (gen) | Survival ON-OFF | Energy ON-OFF | Eat-fraction ON-OFF | Memory strength |wh| ON / OFF | Signal share ON / OFF |
|---|---|---|---|---|---|
| 61 / 61 (same tick, same season phase) | -10.4 | +0.08 | -0.118 | 0.269 / 0.272 | 15.3% / 10.1% |
| 75 / 77 (both tick 18000, phase matched) | -5.5 | +0.38 | +0.032 | 0.299 / 0.307 | 8.9% / 13.0% |

First reading: no sign memory is being used. Memory strength in ON matches the
neutral-drift baseline (both grew from ~0.03 to ~0.27 by mutation alone), so
selection is neither favouring nor removing it. ON survives worse at this
checkpoint. Possibly memory adds instability: a carried-over state can keep
pushing an agent toward an action after its situation has changed. Agents
with memory also signal more (15% vs 10%), worth watching. Both arms were
tested at tick 14000, so the season phase is matched in this reading.

Second reading (tick 18000, phase matched): ON again survives worse (-5.5),
and memory strength stays at the neutral-drift baseline (0.299 vs 0.307). The
signalling difference from the first reading reversed, so it was noise.

Verdict: REJECTED. In two phase-matched readings, memory did not improve
survival and selection did not strengthen the recurrent weights beyond random
drift, so evolution is not putting this memory to use. Likely reasons: (1) a
plain recurrent state with no gating can carry stale information that pushes
an agent toward an action after its situation changes; (2) the season period
(600 ticks) may be too slow for a 24-unit state that mostly decays within a few
ticks, so there is little the memory could usefully hold. Memory stays OFF in
later stages. Worth retrying later with a different design: a gated memory
(GRU-style), a slow-decaying "season clock" unit, or shorter, faster cycles.

## Stage 5: hazards (poison cells, ~3% of grid, drain 0.15 energy/tick, relocate each season)
Arms: hazards_on_seed1 vs hazards_off_seed1 (code: lab/stages/05_hazards).
Social sensing, communication, seasons on; memory off. Both arms have a 5x5
hazard-sensing patch (reads zero in OFF).

Avoidance (ON arm; share of agent-ticks on poison vs poison cover; 1.0 = random):

| Tick (gen) | Agents sampled | Time on poison | Poison cover | Ratio |
|---|---|---|---|---|
| 2500 (~15) | 31 (population was tiny) | 3.09% | 3.52% | 0.88 |
| 10000 (~42) | 100 | 2.10% | 3.04% | 0.69 |
| 16000 (~64) | 100 | 1.63% | 3.21% | 0.51 |
| 23500 (~97) | 100 | 1.74% | 2.60% | 0.67 |

Early read: avoidance is emerging. By gen ~42 agents spend about 31% less time
on poison than chance would give. The ON population survived an early crash
(24 agents at tick 650 in the smoke test) and has recovered to ~870.

On/off comparison:

| Check (gen) | Survival ON-OFF | Energy ON-OFF | Population ON / OFF | Move share ON / OFF | Signal share ON / OFF |
|---|---|---|---|---|---|
| 59 / 61 (ticks 15000 vs 13500: OPPOSITE season phases) | -48.3 | +2.34 | 531 / 1462 | 12.8% / 9.6% | 11.7% / 7.7% |
| 59 / 65 (both tick 15000, phase matched) | +4.2 | +0.47 | 531 / 1018 | 12.9% / 14.6% | 11.8% / 11.1% |

This reading's survival and energy are not trustworthy: the two checkpoints sit
half a season cycle apart, so OFF was tested in plenty (all 50 survived all 150
steps) and ON going into lean. Behaviour is the usable part: with poison present,
agents move more and signal more. Next reading should compare the same tick in
both arms (e.g. 20000 vs 20000), which matches season phase.

Phase-matched reading (tick 15000 both): individual agents in the poison world
do as well or slightly better (+4.2 survival, +0.47 energy, +12 points more
agents gaining energy). The earlier -48 was the season mismatch, and so were
the earlier "moves more / signals more" differences, which vanished once phases
matched. Population after tick 5000: ON 300-1509 (mean 782), OFF 513-1685
(mean 1084), so poison costs about 28% of the population but never collapsed it.

Verdict: ADOPTED as an environment. Sustainable, and agents evolved to sense
and avoid poison (time on poison 0.51-0.69 of chance after gen ~40, from 0.88
early), with no rule telling them to. Hazards stay on in later stages.

## Stage 6: kin recognition (heritable family tags, 5x5 relatedness patch) + share action
Arms: kin_on_seed1 vs kin_off_seed1 (code: lab/stages/06_kin). Social sensing,
communication, seasons and hazards on; memory off. The share action (give 0.5
energy to a random neighbour, plus 0.02 overhead) exists in both arms; only
whether agents can see relatedness differs.

Setup note: started from random brains, this stage went extinct three times
(80 founders; 200 founders, which starved faster; 80 founders with a 3000-tick
grace period). Likely cause: random brains share about 1 action in 12, which
spreads energy thinly so few agents reach the energy needed to reproduce, on
top of seasons and poison. Fix: both arms are seeded with the same 300 evolved
agents from stage 5 (kin inputs wired at zero, share starting rare). Both arms
then held 500-650 agents through a 3600-tick test.

| Check (gen) | Survival ON-OFF | Energy ON-OFF | Population ON / OFF | Share action ON / OFF | Kin share ratio ON / OFF |
|---|---|---|---|---|---|
| 58 / 55 (both tick 12500, phase matched) | -9.9 | -0.56 | 1255 / 1262 | ~0.2% / ~0.1% | 1.26 / 0.94 |

Sharing was almost eliminated by selection in both arms. The faint kin
preference in ON (1.26) rests on very few share events. Diagnosis: a DESIGN
FLAW. A gift of 0.5 energy cost the giver 0.52, so benefit never exceeded cost.
Hamilton's rule says helping relatives can evolve only when r*b > c
(relatedness x benefit to receiver > cost to giver); with b <= c that fails
for every r < 1, so sharing could never pay, kin or not. Selection removed it,
correctly.

Verdict: INCONCLUSIVE (design flaw), not adopted. Kin perception did not help
survival at this reading, but the stage could not test kin altruism. Re-run as
06b with productive sharing (receiver gets 2x what the giver gives up, b/c = 2,
like food being worth more to a hungry agent), seeded from the same stage-5
founders.

## Stage 6b: kin recognition with productive sharing (receiver gets 2x what the giver gives)
Arms: kin_on_seed1 vs kin_off_seed1 (code: lab/stages/06b_kin). Same seeding as 06.

| Check (gen) | Survival ON-OFF | Energy ON-OFF | Population ON / OFF | Share action ON / OFF | Kin share ratio ON / OFF | Recipient vs neighbour relatedness ON ; OFF |
|---|---|---|---|---|---|---|
| 55 / 55 (both tick 12500, phase matched) | +4.5 | -0.12 | 1383 / 1322 | 0.06% / 0.04% | 1.39 / 0.80 | 0.404 vs 0.370 ; 0.255 vs 0.279 |
| 63 / 62 (both tick 15000, phase matched) | -7.5 | -0.34 | 769 / 795 | 0.06% / 0.04% | 0.85 / 1.00 | 0.409 vs 0.404 ; 0.367 vs 0.280 |

First reading: sharing is still rare even though gifts are now productive.
Likely reason: gifts go to a random neighbour, and average neighbour similarity
(~0.37) is below the ~0.52 a gift needs to pay off under Hamilton's rule
(cost 0.52 / benefit 1.0), so most gifts still lose. But the kin signal now
points the same way twice (stage 06: 1.26 vs 0.94; 6b: 1.39 vs 0.80): agents
that can see relatedness share more when relatives are near, and their
recipients are more related than average; blind agents show no such bias.
Small numbers (roughly 100 share events per arm), so suggestive only.
Novelty meter: action mix still shifting (~0.02-0.04 per 1000-tick window, not
flat). Blank entries come from restarts, which reset the comparison window.

Second reading reversed both signals: survival -7.5 (from +4.5) and kin ratio
0.85 vs 1.00 (from 1.39 vs 0.80). The "recipients more related than average"
pattern even appeared in the BLIND arm (0.367 vs 0.280), so it comes from
relatives clustering (viscous population), not from choice. With ~100 share
events per reading, the earlier kin bias was within noise.

Verdict: NOT ADOPTED. Kin perception gave no consistent survival benefit, and
no robust kin favouritism appeared. Sharing stayed rare (~0.05%) even when
productive, most likely because gifts go to a random neighbour whose average
relatedness (~0.3-0.4) is below the ~0.52 needed for a gift to pay off.
A design that lets the giver CHOOSE the recipient (e.g. most related
neighbour) would be the natural retry; noted for later, not done now.
Population after tick 3000: ON 596-1746, OFF 594-1886 (no collapse either way).

## Stage 7: two-parent (sexual) reproduction
Arms: sexual_on_seed1 vs sexual_off_seed1 (code: lab/stages/07_sexual). Social
sensing, communication, seasons and hazards on; memory and kin perception off.
Founders: the same 300 evolved stage-5 agents in both arms. A "mate" action
exists in both arms: the initiator pays exactly the asexual reproduction cost,
the partner (a random neighbour) pays nothing, and the child's genome is a
uniform crossover of both parents plus mutation. In OFF it fails. So the only
difference from asexual reproduction is genetic mixing. Smoke test: both arms
~1,400 agents at tick 1500; ON had 188 two-parent births out of 4,838.
Key measure later: does the share of two-parent births grow (selection favouring
mixing) and does ON survive better?

Interim (to gen ~43): share of births that are two-parent in the ON arm stays
flat at 3.4-4.5% per 1500-tick window, no upward trend yet, so selection is
not (yet) favouring genetic mixing. births.csv was deduplicated for this count
because bit-exact replays after restarts re-log the same births.

| Check (gen) | Survival ON-OFF | Energy ON-OFF | Population ON / OFF | Two-parent share of births (ON) |
|---|---|---|---|---|
| 54 / 51 (both tick 12000, phase matched) | -1.0 | -0.25 | 831 / 830 | 3.4-4.5% early windows, 5.9% and 4.9% in the latest two |
| 63 / 61 (both tick 14500, phase matched) | -14.8 | -0.61 | 779 / 749 | 4.9% (latest window) |

First reading: no survival difference (-1.0 of 150 is within noise). Mating
stays a minor route (about 1 birth in 20), with perhaps a slight late rise
that needs the next reading to confirm.

Verdict: NOT ADOPTED. Two-parent reproduction did not help (-1.0, then -14.8
survival) and selection barely increased its use (about 4% of births rising to
about 5%). Likely reason, well known in neuroevolution as the "competing
conventions" problem: uniform crossover between two unrelated networks mixes
weights that only worked in their original combination, so many two-parent
children are worse than either parent. Mates here are random neighbours.
Retry ideas for later: choose mates by similarity (e.g. close family tags) or
cross over whole hidden units instead of single weights.
Population after tick 3000: ON 565-1708, OFF 584-1667 (no collapse).

## Stage 8: predators (ENVIRONMENT; scripted, non-evolving)
Arms: predators_on_seed1 vs predators_off_seed1 (code: lab/stages/08_predators).
Social sensing, communication, seasons, hazards on; memory, kin perception and
sexual reproduction off. Founders: same 300 evolved stage-5 agents in both arms
(predator input wired at zero). Both arms have a 5x5 predator-sensing patch.

Tuning: first version (1 predator per 100 cells, could eat every tick) drove
the world extinct by tick 337. Second (1 per 200 cells, 15-tick digesting pause
after a kill) survived but fell to 2 agents during the founders' boom/bust.
Final: same, plus no hunting before tick 1500 and full hunting by tick 3000.
4000-tick test: lowest population after hunting starts 495, 1,495 kills,
1,396 agents at the end. Key measure later: do agents evolve to avoid
predators (time within 1 cell of a predator vs chance)?

Predator avoidance (ON arm; whole world run 60 ticks from the checkpoint;
agent-ticks within 1 cell of a predator vs cells within 1 cell of a predator):

| Tick (gen) | Agents near a predator | Cells near a predator | Ratio |
|---|---|---|---|
| 2500 (~16, hunting just starting) | 4.21% | 3.99% | 1.05 |
| 7500 (~37) | 4.04% | 4.21% | 0.96 |
| 12500 (~57) | 4.30% | 4.25% | 1.01 |
| 12500 CONTROL: predator-free arm's agents dropped into the same predator world | 4.25% | 4.28% | 0.99 |
| 15000 (~68) own agents / CONTROL | | | 1.03 / 1.08 (relative 0.95) |

No avoidance yet. Note the baseline is conservative: predators actively chase
agents, so a predator-blind population would sit ABOVE 1.0; real avoidance has
to beat that pull.

On/off comparison:

| Check (gen) | Survival ON-OFF* | Energy ON-OFF* | Population mean after tick 3000 ON / OFF | Kills so far |
|---|---|---|---|---|
| 57 / 52 (both tick 12500) | +10.7 | +0.35 | 1075 / 1130 | 7,187 |
| 68 / 62 (both tick 15000) | +1.3 | -0.05 | 1091 / 1141 | |

*The evaluation tool moves agents but not predators, so these two columns
measure foraging, not escaping predation: agents from the predator world forage
better. One possible reason: predators thin out crowds, so survivors face less
competition at food; another is just noise. Untested.

First reading: no predator avoidance. Agents evolved with predators end up near
predators exactly as often as predator-naive agents in the same predator world
(1.01 vs 0.99). Likely reason: predation pressure is mild (about 0.7 kills per
tick against ~1,000 agents, so roughly 1 in 6 deaths), too weak to select for
avoidance at this depth. Population cost is small (about 5%).

Second reading (tick 15000): avoidance relative to the naive control 0.95
(from 1.02), far from "well below 1"; the foraging edge shrank to +1.3; agents
in the predator world moved more at this reading (15.1% of actions vs 10.3%),
measured only once. Population cost stayed ~4-5%, lowest 509, no collapse.

Verdict: NOT ADOPTED. Sustainable, but no consistent behavioural response at
this predation level (about 0.7 kills per tick, roughly 1 in 6 deaths), so the
base world stays simpler for later stages. Retry idea: stronger or faster
predators, or predators that evolve, so that avoiding them matters more.

## Stage 9: world markers / external memory
Arms: markers_on_seed1 vs markers_off_seed1 (code: lab/stages/09_markers).
Social sensing, communication, seasons, hazards on; memory, kin perception,
sexual reproduction and predators off. Founders: the same 300 evolved stage-5
agents (new marker inputs at zero; "mark" starts rare). A "mark" action adds 1.0
to the agent's cell; marks halve every 200 ticks; agents sense a 5x5 marker
patch (zeroed in OFF; marking still allowed and costed there). Smoke test: both
arms ~1,500 agents at tick 1500, ~1,500 marked cells in each.
Key measures later (lab/marker_analysis.py): are marks laid on richer food than
unmarked cells, and do agents move toward marks more than the 25% chance rate?

| Tick (gen) | Food on marked / unmarked cells, ON | same, OFF | Moves toward strongest mark ON / OFF (chance 25%) | Mark action share ON / OFF |
|---|---|---|---|---|
| 5000 (~28 / 25), early | 0.214 / 0.208 | 0.201 / 0.217 | 25.0% / 24.4% | 0.36% / 0.29% |
| 12500 (60 / 57) | 0.032 / 0.031 | 0.033 / 0.032 | 25.8% / 25.6% | 0.23% / 0.41% |
| 15000 (69 / 65) | 0.176 / 0.170 | 0.153 / 0.161 | 25.3% / 24.2% | 0.26% / 0.46% |

Early read: no trail behaviour yet. Marks sit on average food in both arms and
nobody follows them above chance. Marking is rare (~0.3% of actions).

First full reading (tick 12500, gen 60 / 57; a lean point in the season cycle,
hence the low food values): marks still carry no information. They sit on
average food, and agents that can sense them follow them no more than blind
agents do (25.8% vs 25.6%, chance 25%). Survival ON-OFF -7.1, energy -0.04,
population 1286 vs 1416. Marking is rare in both (0.2-0.4%).

Second reading (tick 15000): survival ON-OFF +4.4 (flipped from -7.1), energy
+0.87; marks still sit on almost average food (0.176 vs 0.170) and are followed
at chance (25.3%, vs 24.2% in the blind control).

Verdict: NOT ADOPTED. No consistent survival effect and no sign that marks carry
meaning or that agents use them: no trails formed. Likely reasons: marking
starts rare and costs energy, a single fading number per cell is a weak signal,
and nothing ties marking to food at the start, so there is no foothold for
selection. Retry idea: let marks carry the sender's food level (like stage 2's
symbols, which did evolve meaning), or make marks cheaper.

## Stage 10: lifetime learning
Arms: learning_on_seed1 vs learning_off_seed1 (code: lab/stages/10_learning).
Social sensing, communication, seasons, hazards on; memory, kin perception,
sexual reproduction, predators and markers off. Founders: same 300 evolved
stage-5 agents, each with a heritable learning-rate gene of 0.01 (mutates in
offspring, x exp(N(0, 0.2))). In ON, after every action an agent adds
lr x (energy change this tick) x (hidden activity) to that action's output
weights in its own lifetime copy; children inherit only the genome. In OFF the
gene exists but is never used, so its drift is the neutral baseline.
Smoke test (1500 ticks): both ~1,420 agents; ON agents' weights changed by
~0.004 on average during life, OFF none; mean lr 0.0111 ON vs 0.0115 OFF.
Key measures: survival, and whether mean lr rises in ON above OFF's drift
(evolution favouring learners).

Interim learning-rate gene (mean / median):

| Tick (gen) | ON | OFF (neutral drift) | ON: mean learned change in weights |
|---|---|---|---|
| 2500 (~15) | 0.0121 / 0.0100 | 0.0129 / 0.0104 | 0.0045 |
| 4500 vs 5000 (~24-25) | 0.0126 / 0.0099 | 0.0141 / 0.0092 | 0.0046 |
| 12500 both (58 / 55) | 0.0129 / 0.0100 | 0.0183 / 0.0136 | 0.0061 |
| 15000 both (69 / 63) | 0.0132 / 0.0085 | 0.0193 / 0.0124 | 0.0057 |

No sign yet that selection favours learners: the gene drifts the same in both.

| Check (gen) | Survival ON-OFF | Energy ON-OFF | Share improving ON-OFF | Population ON / OFF |
|---|---|---|---|---|
| 58 / 55 (both tick 12500) | -8.4 | +0.24 | +12 points | 1253 / 1372 |
| 69 / 63 (both tick 15000) | +13.2 | +0.53 | +8 points | 738 / 821 |

First reading: learners survive worse (-8.4) though slightly more of them gain
energy. The learning-rate gene is now LOWER where it is used (mean 0.0129, median
0.0100, i.e. stuck at the starting value) than in OFF, where it drifts freely
upward (0.0183). That suggests selection is holding learning down: agents that
learn faster do worse, consistent with a noisy reward signal (energy change per
tick also reflects seasons and luck) teaching the wrong lessons.

Verdict: NOT ADOPTED. Survival flipped (-8.4, then +13.2), so the rule is not
met. Two things were consistent across both readings: learners gained more
energy (+0.24, +0.53) and more of them improved (+12, +8 points); and the
learning-rate gene stayed low where it was used (median at or below its 0.01
start) while drifting up in OFF. So a little learning may help, but selection
works against learning MORE. Likely reason: the teaching signal (energy change
this tick) is noisy, since it also moves with seasons, poison and luck.
Retry idea: learn from a cleaner signal (food actually eaten), or let the
rate decay with age.

## Stage 11: self-prediction (marker for self-modelling, NOT a test of consciousness)
Arms: selfpred_on_seed1 vs selfpred_off_seed1 (code: lab/stages/11_selfpred).
Social sensing, communication, seasons, hazards on; everything rejected so far
off. Founders: same 300 evolved stage-5 agents, each given a small random
prediction head on the hidden layer. Every tick each agent predicts its own
next-tick energy (fraction of max). ON pays 0.03 energy per unit of error
(error is 0..1; a move costs 0.03); OFF makes the prediction with no cost, so
its accuracy is the neutral baseline.
Smoke test (1500 ticks): mean lifetime error 0.276 ON vs 0.286 OFF; population
1,053 ON vs 1,446 OFF (being wrong is costly at first).
Caveat, stated up front: energy is already an input, so the easy solution is
"my energy stays about the same". Falling error mainly shows agents can read
their own state; predicting the CHANGE in energy would be a stronger test.
Key measures: mean lifetime prediction error per arm over generations, and survival.

### Stage 11 interim (08:56 UTC, same-tick checkpoints, seed 1; not a verdict, below gen 55)
| tick | ON pop | OFF pop | ON gen | OFF gen | ON mean pred. error | OFF mean pred. error |
|---|---|---|---|---|---|---|
| 2500 | 357 | 631 | 10.0 | 9.7 | 0.243 | 0.322 |
| 5000 | 719 | 913 | 21.3 | 20.5 | 0.206 | 0.291 |
| 7500 | 1151 | 1425 | 31.1 | 29.9 | 0.200 | 0.303 |
| 10000 | 1211 | 1648 | 40.3 | 40.0 | 0.201 | 0.299 |
| 12500 | 763 | 1411 | 49.2 | 49.2 | 0.211 | 0.318 |
Prediction error falls in ON and stays ~0.30 in OFF: selection does shape the self-prediction head when it is costly. ON population runs 20-40% below OFF at every reading (the error cost is a tax). ON dipped to 377 at tick 12000 (seasonal-looking, recovered to 763 by 12500); watch for collapse. Verdict waits for OFF to reach 12500/15000.

### Stage 11 reading 1 (tick 12500, both arms mean gen 49.2, evaluate.profile 50 agents x 150 steps)
| metric | OFF | ON | ON-OFF |
|---|---|---|---|
| eat-fraction | 0.855 | 0.776 | -0.079 |
| energy change | 1.463 | 0.542 | -0.921 |
| steps survived /150 | 119.4 | 92.7 | -26.7 |
| share improving | 0.62 | 0.48 | -0.14 |
| population | 1411 | 763 | -648 |
Clear ON deficit on every survival metric. Second reading at tick 15000 decides.

### Stage 11 reading 2 (tick 15000, ON gen 57.9 / OFF gen 58.9)
| metric | OFF | ON | ON-OFF |
|---|---|---|---|
| eat-fraction | 0.769 | 0.577 | -0.192 |
| energy change | 1.993 | 1.834 | -0.159 |
| steps survived /150 | 93.7 | 88.8 | -4.9 |
| share improving | 0.76 | 0.80 | +0.04 |
| population | 853 | 384 | -469 |
| mean prediction error | 0.292 | 0.217 | -0.075 |

**Stage 11 verdict: NOT ADOPTED.** ON lost on survival in both readings (-26.7, -4.9) and ran at 45-55% of OFF's population throughout, with no extinction. Positive capability result: when mispredicting costs energy, selection drives self-prediction error down (~0.21 vs ~0.30 at every reading from tick 2500 on); the self-model is learned but does not pay for itself here. Caveat from design stands: predicting next energy is easy given energy is an input; a stronger test would predict the change, or make the prediction an input the agent can act on. Self-prediction is OFF in both arms of stage 12. Tally: 4 of 13 tested features kept.

## Stage 12: richer language (two-symbol messages), CAPABILITY stage
Design: built from stage 11's code, self-prediction OFF in both arms (not adopted). Same 300 stage-5 founders in both arms (seeded surgery: 25 zero input rows for a second-symbol patch, a near-uniform second-symbol head ws/bs). Whenever an agent signals (signal_N action, same 0.015 cost as before) it also picks a second symbol 1-4 from the new head, so a message is one of 16. ON: receivers sense both symbols (signal + signal2 patches). OFF: identical sender behaviour and cost, but only the first symbol is delivered (signal2 grid stays zero). Measure (lab/language_analysis.py): from each agent's brain in its actual situation, I(situation; sym2 | sym1) in bits, situations = food direction (6 classes) and energy (2), minus a shuffled-label baseline; ON vs OFF. Also survival via evaluate.profile. Smoke test (tick 500, seed 77): second-symbol use uniform (0.25 each), excess information ~0 in both arms, as it should be at the start.
Caveat: four first symbols already exist and carried little information by the end of stage 2, so the second symbol may be a channel with nothing to say; a null result here is informative, not a failure of the code.

### Stage 12 interim: second-symbol excess information, I(s;sym2|sym1) minus shuffled baseline (bits; food direction / energy)
| tick | ON | OFF |
|---|---|---|
| 2500 | -0.0006 / +0.0001 | +0.0045 / -0.0007 |
| 5000 | +0.0025 / -0.0011 | -0.0029 / -0.0004 |
| 7500 | +0.0034 / -0.0003 | +0.0092 / +0.0006 |
| 10000 | -0.0013 / +0.0037 | +0.0032 / -0.0030 |
| 12500 | -0.0047 / -0.0017 | -0.0090 / +0.0031 |

### Stage 12 reading 1 (tick 12500; mean gen ON 49.7, OFF 45.6; evaluate.profile 50 agents x 150 steps)
| metric | OFF | ON | ON-OFF |
|---|---|---|---|
| eat-fraction | 0.827 | 0.782 | -0.045 |
| energy change | 1.213 | 1.270 | +0.057 |
| steps survived /150 | 112.9 | 112.0 | -0.9 |
| share improving | 0.62 | 0.62 | 0.00 |
| population | 1355 | 1375 | +20 |
Survival: a tie. Second symbol: no information in either arm at any reading (all within ±0.01 bit of the shuffled baseline; ON never above OFF consistently). Second reading at tick 15000.

### Stage 12 reading 2 (tick 15000; mean gen ON 59.1, OFF 55.1)
| metric | OFF | ON | ON-OFF |
|---|---|---|---|
| eat-fraction | 0.683 | 0.679 | -0.004 |
| energy change | 1.690 | 2.068 | +0.378 |
| steps survived /150 | 91.0 | 99.2 | +8.3 |
| share improving | 0.70 | 0.74 | +0.04 |
| population | 784 | 800 | -16 |
| 2nd-symbol excess info, food dir / energy (bits) | +0.0030 / +0.0040 | -0.0039 / +0.0011 | |

**Stage 12 verdict: NOT ADOPTED.** Survival matched (reading 1 -0.9) then favoured ON (+8.3), with no collapse, but the second symbol carried no information about the sender's situation in either arm at any of six readings (all within ±0.01 bit of the shuffled baseline, ON never consistently above OFF). Without information in the channel, the reading-2 survival gap cannot be credited to language; it is most likely season-phase/sampling noise (ON is also 4 generations ahead). Only change in ON: second-symbol use drifted from uniform (0.14/0.26/0.27/0.33), a bias with no meaning. Consistent with stage 2, where first symbols carried little: in this world there is little worth saying. Retry idea: a world where information is valuable and not directly sensed (food beyond sense radius, a hidden shared-food location). Language stays OFF in stage 13. Tally: 4 of 14 tested features kept.

## Stage 13: shared food (ENVIRONMENT stage, last on the roadmap)
Design: built from stage 12's code, language and self-prediction OFF in both arms (not adopted). Same 300 stage-5 founders in both arms (25 zero input rows for a shared-food patch). Each tick, with probability 0.5, 4.0 energy enters the world. ON: as one shared-food item in a random cell, lasting 150 ticks, that can only be eaten when 2+ agents use "eat" in that cell on the same tick (they split it; a lone eater gets nothing from it). OFF: the same 4.0 energy is added as ordinary food in random cells, so energy input is identical; only the need to eat together differs. Logged per run in cofood.csv (cumulative spawned / eaten / expired / solo_attempts / co_eaters).
Measure: survival ON vs OFF (evaluate.profile), and in ON the trend in the share of shared food eaten vs expired, and in wasted solo attempts per spawn, over generations.
Caveat: the smoke test showed founders already co-eat by chance because they crowd around food (seed 77, ~800 ticks: 290 of 395 items eaten jointly, with 3330 wasted solo attempts). So "cooperation evolves" has to show up as a change from that starting level, not as co-eating existing at all.

### Stage 13 shared-food trend (ON arm, per 1000-tick window)
| ticks | eaten | expired | wasted solo attempts / item | mean pop |
|---|---|---|---|---|
| 0-1000 | 78% | 16% | 9.4 | 1171 |
| 1000-2000 | 73% | 23% | 10.4 | 1088 |
| 2000-3000 | 82% | 15% | 9.8 | 1257 |
| 3000-4000 | 81% | 23% | 11.4 | 1147 |
| 4000-5000 | 74% | 18% | 10.5 | 1109 |
| 5000-6000 | 87% | 13% | 9.9 | 1257 |
| 6000-7000 | 87% | 16% | 11.0 | 1208 |
| 7000-8000 | 77% | 18% | 11.2 | 1116 |
| 8000-9000 | 89% | 11% | 10.4 | 1256 |
| 9000-10000 | 86% | 16% | 11.7 | 1211 |
| 10000-11000 | 84% | 12% | 11.0 | 1216 |
| 11000-12000 | 91% | 8% | 10.2 | 1408 |
| 12000-13000 | 89% | 13% | 11.4 | 1278 |
| 13000-14000 | 86% | 9% | 10.6 | 1256 |
| 14000-15000 | 93% | 9% | 11.3 | 1391 |
Density check: eaten share correlates with window population r = 0.88; with time r = 0.69; after removing the population effect, the time trend is weak (r = 0.35, 13 windows). Wasted solo attempts do not fall. So the rise in eaten share is mostly "more agents, more accidental co-eating", with at most a weak residual that could be cooperation.

### Stage 13 reading 1 (tick 12500; mean gen ON 46.8, OFF 49.4)
| metric | OFF | ON | ON-OFF |
|---|---|---|---|
| eat-fraction | 0.832 | 0.870 | +0.039 |
| energy change | 0.958 | 1.465 | +0.506 |
| steps survived /150 | 116.1 | 127.6 | +11.5 |
| share improving | 0.56 | 0.64 | +0.08 |
| population | 1423 | 1480 | +57 |
ON ahead on every survival metric. Note the 150-step test runs each agent alone, so shared food (needs 2 eaters) cannot be eaten in it; ON's advantage is in general foraging, perhaps selection for staying close to food and others. Second reading at tick 15000.
Checked: compare.run_agent steps one agent alone in a copy of its world and never calls resolve_cofood, so in the test ON agents cannot get shared food at all, while OFF agents still receive the equivalent energy as ordinary food. The test therefore favours OFF, which makes ON's +11.5 lead more notable, not less.

### Stage 13 reading 2 (tick 15000; mean gen ON 55.2, OFF 58.3)
| metric | OFF | ON | ON-OFF |
|---|---|---|---|
| eat-fraction | 0.802 | 0.828 | +0.026 |
| energy change | 2.084 | 2.379 | +0.296 |
| steps survived /150 | 98.8 | 104.0 | +5.2 |
| share improving | 0.82 | 0.86 | +0.04 |
| population | 906 | 913 | +7 |
Density check over 15 windows: eaten share vs population r = 0.90; after removing population, time trend r = 0.29 (weak). Wasted solo attempts per item 9.9 early (0-3000) vs 11.1 late; per 1000 agents 8.47 vs 8.50, i.e. unchanged.

**Stage 13 verdict: NOT ADOPTED as a cooperation feature; environment effect noted.** Survival favoured ON in both readings (+11.5, +5.2) with no collapse, even though the solo survival test gives ON agents no access to shared food. But cooperative eating did not clearly improve: the rise in eaten share (~78% to ~90%) is almost entirely explained by population density, and wasted solo attempts per agent did not fall. So the shared-food world helps survival (likely by selecting for staying near food and near others), but there is no evidence that agents learned to coordinate. That survival effect is the most promising single-seed result since hazards and should be the first thing replicated on seeds 2-5. Final tally on the roadmap: 4 of 15 tested features kept (social sensing, communication, seasons, hazards), plus one promising environment effect (shared food) awaiting replication.


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


## Sparse world, first experiments (24 Nov 2026, fast engine)

World: 128 x 128 cells; food regrows only inside 16 drifting clusters
(Gaussian, width 4 cells, peak regrowth 0.057 per tick, centres moving 0.03
cells/tick on slowly turning headings); bare ground between clusters;
sensing radius 2 (so food is usually out of sight). Social sensing and
communication on, no seasons or hazards. Founders: the 300 evolved stage-5
agents, placed near the clusters. 25,000 ticks, seeds 1-5, ON and OFF arms
(30 runs). The world is sustainable: populations settle around 2,700-3,200,
no run died out, and agents follow the moving clusters (median distance to a
cluster centre falls from ~9 to ~7 cells in the first 5,000 ticks).
Code: fast/engine.py (food_mode="patches"), fast/modes.py (sparse_*),
fast/analyze_sparse.py.

New in the engine: working memory (the hidden state carried to the next tick
through wh, as in old stage 4), checked to give identical results in the
NumPy and compiled paths.

### Results (lesion test: the ON arm's own agents, own world, equal energy/age, with vs without the feature)
| Experiment | Effect of keeping the feature | Population after tick 15k, ON vs OFF |
|---|---|---|
| Working memory | energy +0.00 (better on 4/10 readings); eat-fraction -0.07, lower on 10/10 | 2,668-2,938 vs 2,831-3,129 |
| Communication | energy -0.18 (better on 2/10); eat-fraction -0.02 | 2,831-3,129 vs 2,931-3,149 |
| Shared food | (no lesion) eaten share 0.51-0.65 -> 0.82-0.84 and expired 0.32-0.46 -> 0.15-0.17 on 5/5 seeds, but eaten share tracks population (r = 0.97-0.98; no trend left after removing it). Wasted solo attempts per item per 1,000 agents fell on 5/5 (about 4.6 -> 3.1): a weak hint, also explainable by faster eating in a busier world. | 3,031-3,195 vs 3,019-3,269 |

So far, even where food is out of sight, memory changes behaviour (memory
users eat less often) without paying off, signals are not used, and shared
food shows no clear coordination. Note this is after 25,000 ticks (about
100 generations) from founders adapted to a dense world.

### Method note: the same-world "fair test" is invalid in the sparse world
Tested agents keep their positions from their own run. In the sparse world
the two runs' clusters have drifted apart, so agents tested in their own
world sit on food and agents moved to the other world do not: every
same-world test came out "home world wins 5/5" (memory -1.40 / +1.03,
communication -1.45 / +1.66, shared food -0.98 / +1.22). Those numbers are
not used. The same effect may partly affect the uniform-world same-world
tests above (agents sit where their own world's food was). The lesion test
(same agents, same world, same positions) does not have this problem and is
the basis for verdicts. A position-matched fair test (both arms placed on
the same sampled cells) is the fix if needed.

### Next ideas for the sparse world
* Run longer (100k+ ticks) so evolution can adapt to the new world.
* Make information more valuable: faster-moving or short-lived clusters,
  fewer clusters, larger gaps; or rich clusters that appear and vanish, so
  remembering or hearing where food was matters.
* Shared food that is worth more and lasts briefly, so two agents must
  arrive together (a real coordination problem).

## Step 1 complete: fast engines (24 Nov 2026)
* fast/engine.py (NumPy + Numba): ~110x the old engine at ~1,500 agents on 2 cores (6.1 -> 674 ticks/s,
  same machine and state); exact match with its own NumPy path; statistical match with the old engine
  (stage 13 over 15,000 ticks; stages 1, 2, 3, 5 population histories on seeds 1-5).
* fast/engine_torch.py (PyTorch, GPU): same rules, same checkpoint format; matches the numba engine
  statistically (stage 13 and sparse worlds, 5 seeds each).
* Laptop benchmark (128 x 128 world, ~10,400 agents; bench_gpu.py):
  numba CPU, 16 threads: 160 ticks/s; torch on RTX 5060 Laptop GPU: 110 ticks/s; old engine
  (estimated from its 6.1 ticks/s at 1,500 agents): ~0.9 ticks/s. So ~180x on the CPU engine.
  The GPU engine is slower here because each tick still syncs with the CPU several times (sharing,
  shared food, bookkeeping); it should overtake at much larger populations or once those steps move
  to the GPU. For now the numba engine is the one to use.
* Added for steps 2-3: working memory, two-symbol language, sparse world v1 and v2 (vanishing
  clusters), novelty meter (novelty.csv) and family trees (births.csv), both engines.


## Sparse world v2: 100,000-tick runs (24 Nov 2026, laptop, fast engine)

World v2: 128 x 128; 8 richer food clusters (same total food as v1) drifting
at 0.05 cells/tick, each vanishing at random (mean life 400 ticks) and
reappearing elsewhere; shared food worth 8 energy, lasting 30 ticks, needing
2 eaters on the same tick. Founders: the 300 stage-5 agents near the clusters.
4 experiments x ON/OFF x seeds 1-5, 100,000 ticks each (about 400
generations), 40 runs on the laptop (16 threads, ~12 minutes per run, no
extinctions). Logs include the novelty meter (novelty.csv) and family trees
(births.csv). Analysis: fast/analyze_sparse2.py (fast/sparse2_results.md).
Lesion tests: the ON arm's own agents, own world, equal energy (5) and age
(100), with vs without the feature, at ticks 90,000 and 100,000.

| Experiment | Result |
|---|---|
| Working memory | Keeping memory **hurts**: energy -0.50 (better on 1/10 readings), eat-fraction -0.14 (lower on 10/10). Agents that evolved with memory do worse when it is on. Late population ON vs OFF: lower on 3/5 seeds. |
| Communication | Not used: energy -0.00 (better on 5/10), eat-fraction -0.02. |
| Language (2nd symbol) | Not used: energy +0.12 (better on 6/10), eat-fraction -0.01 (higher on 2/10). Information about cluster direction carried by the 2nd symbol: +0.025 bits in ON vs +0.033 in OFF (no more than the control, where it is never heard). |
| Shared food (worth 8, 30 ticks) | **First sign of coordination.** Share eaten rose from 0.15-0.23 (ticks 0-10k) to 0.42-0.60 (90-100k) on 5/5 seeds; expired share fell 0.76-0.84 -> 0.39-0.57. After removing the effect of population size, the rise over time remains on 4/5 seeds (r = +0.71, +0.41, +0.24, +0.25; seed 2: 0.00). Wasted solo attempts per agent fell on 4/5. In world v1 (cheap, long-lasting shared food) the rise was entirely population. Needs a direct test (below). |

Novelty meter: behaviour change per 1,000 ticks falls from ~0.07-0.10
(ticks 2-20k) to ~0.06-0.07 (80-100k) in every experiment: behaviour keeps
shifting a little but is levelling off, with no burst of new behaviour.
Family trees: every run ends with all agents descended from a single
founder (1 of 300 lineages), so diversity comes only from mutation within
that lineage.

Method note: the first language lesion run gave identical numbers with and
without language because the survival test did not feed the second-symbol
inputs. Fixed (fast/evaluate_fast.py now includes them); the table above
uses the fixed test.

### Next
* Shared food: test it directly. Same agents, with vs without the ability to
  see shared food, and both arms' agents placed on the same cells, to check
  that late co-eating is learned coordination, not crowding. Then more seeds
  and longer runs if it holds.
* Memory hurting after 400 generations suggests memory weights inherited
  from memory-free ancestors are noise the agents never clean up; a fairer
  test starts memory from zero weights.
* Language still has nothing to say: messages might need to be cheaper, or
  the world must make a distant observation worth relaying (e.g. clusters
  only one agent can see).

## Step 2 closed: three follow-up tests (24 Nov 2026)

**1. Shared food, direct test.** Same agents, shared food made invisible to them vs visible (5 seeds, sparse world v2).
Blinding changed co-eating by only +0.016 at tick 100k (4/5 seeds) and -0.037 at 10k. The rise in co-eating seen earlier
(0.2 -> 0.42-0.60) is crowding at food clusters, not coordination. Verdict: no coordination.

**2. Fair memory test (memory weights start at zero).** 34_sparse2_memory0, 5 seeds x 2 arms, 100k ticks, all survived.
Lesion on the ON agents (ticks 90k and 100k, memory switched off): keeping memory gives energy -0.23 (better on 4/10)
and eating -0.106 (higher on only 2/10). Memory weights grew from 0 to |w| ~0.82, the same size as unused weights drifting
in the OFF arm (~0.80), so there is no sign selection shaped them. Population ON vs OFF: mixed (ON higher on 3/5).
Verdict: even with a fair start, memory does not help here; the earlier "memory hurts" result was not just bad starting weights.

**3. Terrain + language.** 35_terrain_language: walls every 32 cells (16 rooms, 8-cell doorways), 16 clusters, peak 0.057, life 600.
- Survival: language ON 5/5 seeds survived to 100k; language OFF only 2/5 (seeds 1, 2, 4 died out by ticks 16.7k, 3.9k, 7.4k).
  Suggestive but not significant (one-sided p ~0.08, two-sided ~0.17); needs more seeds.
- Lesion on ON agents (ticks 90k and 100k, language switched off): keeping language gives energy +0.15 (better on 7/10),
  eating +0.002 (higher on 4/10). Small, consistent-ish benefit to energy, none to eating.
- Information carried by the second symbol about cluster direction: +0.003 to +0.034 bits over shuffled (largest seed 5);
  first symbol up to +0.08 bits (seed 4). Weak everywhere; seeds disagree on what is encoded.
Verdict: terrain is the first world where language shows any hint of value (survival 5/5 vs 2/5, +0.15 energy),
but the signals carry very little information. Not yet a real language. Worth more seeds before claiming anything.

**Step 2 summary.** Sparse food and terrain did not make memory or shared food useful. Language moved from "not used"
to "slight hint" once walls made distant information matter. Next in the approved order: step 3 (longer runs, 1M ticks)
and step 5 (more seeds), starting with terrain seeds 6-10 to test the survival hint.
