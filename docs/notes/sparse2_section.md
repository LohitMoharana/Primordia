

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
