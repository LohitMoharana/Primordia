# Lab log

Running notes from the automatic lab (`fast/lab_queue.py` on the laptop). Newest first. Small result
files are copied into `results/` at each update; full runs stay on the laptop.

## 2026-11-26 15:35 IST

**Finished:** remains test (condition 2): no effect (novelty 27.4 vs 24.6, p = 0.51), no scavengers evolved;
both arms far below the plain world (~26 vs ~65). Written into FINDINGS.md. **Running:** fresh-seed check of the
search's worlds (final best, per-creature best, plain world), started 15:31. **Added to the queue:** separate
habitats (no surface food on rich soil + harsh digestion trade-off). Files: `results/remains/`, plus earlier
comparisons copied to `results/` (actions, open_brains2, modes_vs_asal).

## 2026-11-26 14:55 IST

**Finished:** MODES-scored world search, 24 generations (384 worlds, 40,000 ticks each, ~7.5 h). Median
lasting novelty per world rose from 25 (generation 1) to 92 (generation 24); 16/16 worlds survived in each
of the last 13 generations. Best: 153 (generation 22). Full record in `results/modes_search/`.
**Running:** remains test (condition 2), started 14:46.

## 2026-11-26 14:35 IST

**Running:** MODES-scored world search, last generation (24 of 24, ~25 min). Generation 22 set a new best,
**153** lasting new strategies (diversity 3.53 bits, ~6,900 creatures): 14 permanent food clusters, 10 rich
buried-food patches, a little poison, 7 gentle predators, a few shelters. Medians: 98 (gen 22, highest
so far) and 81 (gen 23); 16/16 worlds survived in both. Search ends ~15:00; then the queue runs remains,
the fresh-seed check (final best, per-creature best from gen 6, plain world), shaken, switching, 4x.

## 2026-11-26 13:35 IST

**Running:** MODES-scored world search, generation 22 of 24. Median lasting novelty: 60, 74, **89** in
generations 19-21 (the highest median so far); 15-16 of 16 worlds survive. All-time best still 133.
Three generations left (about an hour): the search should end around 14:30, then the queue runs remains
(condition 2), the fresh-seed check of the search's worlds, shaken, switching and 4x-larger worlds.

## 2026-11-26 12:35 IST

**Running:** MODES-scored world search, generation 19 of 24 (21-23 min per generation). Median lasting
novelty per world: 81, 78, 76 in generations 16-18; every world survived (16/16) for six generations in a
row. All-time best still 133. The search centre has added predators back: 12 permanent food clusters, rich
buried food, a little poison, 15 very gentle predators (catch chance 0.07), no seasons. Expected end:
about 14:30, then the remains test starts.

## 2026-11-26 11:35 IST

**Running:** MODES-scored world search, generation 16 of 24 (20-23 min per generation). Median lasting
novelty per world: 82, 68, 73 in generations 13-15 (it was 25 at the start); every world survived in all
three. All-time best still 133 (generation 11). The search centre has settled on rich, permanent food
clusters, plenty of easily digested buried food, a few shelters, a little poison, and no predators or
seasons. Expected end of the search: about 14:30.

## 2026-11-26 10:35 IST

**Running:** MODES-scored world search (`fast/modes_search.py`), generation 13 of 24, ~20 min per generation.
Median lasting novelty per world has risen from 25 (gen 1) to 68-72 (gens 11-12); best 133.
The newest best has ~7,000 creatures (permanent food clusters, no predators): the raw score is starting
to favour big populations. The best *per creature* is generation 6: 130 new lasting strategies with
1,421 creatures (20 drifting short-lived clusters, mild seasons, rich buried food, 18 gentle predators).
Both, plus the plain world, go into the fresh-seed check.

**Queue:** remains (condition 2) -> fresh-seed check of the search -> shaken world (Rainey & Travisano
1998) -> switching world (Kashtan & Alon 2005) -> 4x larger world (Bejjani et al. 2025).

Files: `results/modes_search/` (every world tried so far, best world, search log).
