# Lab log

Running notes from the automatic lab (`fast/lab_queue.py` on the laptop). Newest first. Small result
files are copied into `results/` at each update; full runs stay on the laptop.

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
