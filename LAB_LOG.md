# Lab log

Running notes from the automatic lab (`fast/lab_queue.py` on the laptop). Newest first. Small result
files are copied into `results/` at each update; full runs stay on the laptop.

## 2026-11-27 11:52 IST

**Finished:** sensitivity check: the best world beats the plain world under all five MODES settings (p = 0.0001),
gap widening with a stricter filter (2.4x / 2.8x / 3.2x); per-creature advantage in 4 of 5 (not with the coarse
skeleton, which saturates). Faster switching (1,000 ticks): no benefit (p = 0.05-0.98). Written into FINDINGS.md.
**Running:** long Bedau run, 1,000,000 ticks, 4 seeds per world, started 11:14. Expected: tomorrow morning.

## 2026-11-27 10:37 IST

**Finished:** 4x world (fixed cap): 6/6 survived with ~27,100 creatures; lasting novelty 255.3 vs 175.7 (1.45x),
diversity 4.23 vs 3.73, but per 1000 creatures 9.4 vs 26.3 (all p <= 0.005). Written into FINDINGS.md.
**Running:** sensitivity check + faster switching (started ~09:05; 30 runs). **Next:** long Bedau run (1M ticks).

## 2026-11-27 07:05 IST

**Running:** 4x world rerun (cap 80,000), 82 min in. The 6 normal-size runs are done (151-201 lasting new
strategies, all alive); the 6 large-world runs are still going, so none has hit a cap this time. Expected
~10:00-11:00.

## 2026-11-27 05:45 IST

**Finished overnight:** switching world (all 30 runs): no extra novelty from switching (it lands between the two
fixed regimes); the harder regime gives the most per creature (39.4 vs 27.3, p = 0.0002) with half the
creatures. Written into FINDINGS.md and README. **4x world: invalid**: every 4x run hit the 20,000 population
cap (my mistake: the cap was set for 1x worlds) and stopped by tick 11,700. **Queued:** the 4x world again with
the cap at 80,000, 6 seeds per arm (needs `python fast/lab_queue.py` running).

## 2026-11-26 22:44 IST

**Running:** switching world, 106 min in, 26 of 30 runs done (all alive). Provisional:
switching every 5000 ticks: novelty 148.1, per 1000 creatures 29.6, diversity 3.65, 5,020 creatures (10 runs);
regime A fixed (the best world): 179.4, 26.7, 3.77, 6,720 (7 runs); regime B fixed (harder: food drifts 3x
faster, 0.6x as rich, predators catch 2.5x as often): 129.8, **40.2**, 3.42, 3,390 (9 runs).
So far switching looks like a mix of A and B, not something beyond both; the harder regime B gives the most
novelty per creature (fewer creatures). Final in the morning. **Next (overnight):** 4x world, ~13:00 tomorrow.

## 2026-11-26 21:44 IST

**Running:** switching world (Kashtan & Alon 2005), 46 min in, 1 of 30 runs done (switching every 5000 ticks:
105 lasting new strategies, 29.5 per 1000 creatures, 3,560 creatures; one run, too early to compare). Expected
end ~00:30-01:30; then the 4x world (~13:00 tomorrow).

## 2026-11-26 21:41 IST

**Finished:** shaken world, all 20 runs: lasting novelty per 1000 creatures 15.3 vs 27.3 (p = 0.0001), total
125.7 vs 182.8 (p = 0.0003), diversity 3.04 vs 3.80 (p = 0.0016). Written into FINDINGS.md and README (✅).
**Running:** switching world (Kashtan & Alon), started 20:57; 1 of 30 runs done.

## 2026-11-26 20:44 IST

**Running:** shaken world, 110 min in; 16 of 20 runs done (all alive), last 4 shaken runs still going.
Provisional (10 vs 6 runs): shaking the best world halves lasting novelty per 1000 creatures (13.9 vs 27.3),
lowers total novelty (113.7 vs 182.8) and diversity (2.75 vs 3.80), all p = 0.0004, although the shaken world
holds more creatures (8,190 vs 6,700). Same direction as Rainey & Travisano 1998. Final numbers in the next entry.

## 2026-11-26 19:45 IST

**Running:** shaken world (Rainey & Travisano), 51 min in, no runs finished yet (all 20 run at once; the
best world holds ~6,700 creatures, so each 100k-tick run takes about 1.5 h). Expected ~21:00. Then the
switching world, then the 4x world overnight.

## 2026-11-26 19:05 IST

**Finished:** separate habitats. Separation: no effect per creature (p = 0.77). Harsher digestion trade-off:
+25% novelty per creature (34.0 vs 27.3, p = 0.0002) but lower total and diversity (smaller population).
Written into FINDINGS.md. **Running:** shaken world (Rainey & Travisano), started 18:53.

## 2026-11-26 18:43 IST

**Running:** separate habitats (93 min in; 20 of 30 runs done, the best-world reference arm still running).
Separate habitats + harsh gut trade-off: lasting novelty 114.4, per 1000 creatures 34.8, diversity 3.20,
3,380 creatures. Overlapping habitats, same trade-off: 143.3, 34.0, 3.25, 4,210 creatures. So far separation
itself makes no difference per creature (it only shrinks the population); the harsher trade-off in both
looks higher per creature than the best world had in the fresh-seed check (27.3), to be confirmed when its
arm finishes.

## 2026-11-26 17:40 IST: first confirmed positive result

**Finished:** fresh-seed check. Both worlds found by the MODES-scored search beat the plain world on 10 fresh
seeds over 100,000 ticks: lasting novelty 182.8 and 114.8 vs 65.6, per 1000 creatures 27.3 and 31.1 vs 20.2,
diversity 3.80 and 3.38 vs 2.75 (all p <= 0.009; 10/10 survived). Written into FINDINGS.md and README.
**Running:** separate habitats (started 17:09).

## 2026-11-26 16:38 IST

**Running:** fresh-seed check (66 min in). The 10 plain-world runs are done and reproduce the earlier numbers
exactly (65.6 lasting new strategies, 20.2 per 1000 creatures; the engine is deterministic per seed). The
search's two worlds (final best, ~7,000 creatures; per-creature best, ~1,400) are still running: bigger
populations run slower. Expected: 17:30-18:00.

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
