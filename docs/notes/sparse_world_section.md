

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
