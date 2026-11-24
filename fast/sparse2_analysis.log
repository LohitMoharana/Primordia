# Sparse world v2: 100,000-tick experiments, seeds 1-5

## sparse2_memory: lesion test (remove 'memory' from the ON arm's own agents)
| seed | energy with / without (90k) | (100k) | eat-fraction with / without (100k) | late population ON / OFF |
|---|---|---|---|---|
| 1 | -0.50 / +0.30 | -0.06 / +0.29 | 0.711 / 0.809 | 1961 / 2129 |
| 2 | -0.62 / -0.60 | -0.86 / -0.17 | 0.612 / 0.882 | 1973 / 2207 |
| 3 | -0.33 / -0.40 | -0.68 / -0.25 | 0.642 / 0.672 | 2154 / 2125 |
| 4 | -0.20 / +0.62 | -0.30 / -0.18 | 0.594 / 0.740 | 2300 / 1824 |
| 5 | -1.06 / -0.03 | -1.27 / -0.41 | 0.563 / 0.681 | 2049 / 2182 |
Keeping memory: energy -0.50 (better on 1/10 readings), eat-fraction -0.144 (higher on 0/10)

## sparse2_comm: lesion test (remove 'communication' from the ON arm's own agents)
| seed | energy with / without (90k) | (100k) | eat-fraction with / without (100k) | late population ON / OFF |
|---|---|---|---|---|
| 1 | -0.11 / -0.30 | -0.53 / -0.32 | 0.706 / 0.685 | 2129 / 2167 |
| 2 | -0.71 / -0.82 | -0.77 / -0.73 | 0.642 / 0.668 | 2207 / 1986 |
| 3 | -0.18 / -0.24 | -0.40 / -0.47 | 0.503 / 0.595 | 2125 / 2345 |
| 4 | -0.51 / -0.29 | -0.60 / -1.11 | 0.661 / 0.636 | 1824 / 2168 |
| 5 | -0.44 / -0.15 | +0.10 / +0.29 | 0.722 / 0.746 | 2182 / 2431 |
Keeping communication: energy -0.00 (better on 5/10 readings), eat-fraction -0.022 (higher on 3/10)

## sparse2_language: lesion test (remove 'language' from the ON arm's own agents)
| seed | energy with / without (90k) | (100k) | eat-fraction with / without (100k) | late population ON / OFF |
|---|---|---|---|---|
| 1 | +0.37 / +0.37 | -0.38 / -0.38 | 0.695 / 0.695 | 2110 / 2129 |
| 2 | -0.02 / -0.02 | +0.55 / +0.55 | 0.835 / 0.835 | 2040 / 2207 |
| 3 | -0.85 / -0.85 | +0.17 / +0.17 | 0.644 / 0.644 | 2320 / 2125 |
| 4 | +0.66 / +0.66 | -0.73 / -0.73 | 0.701 / 0.701 | 2275 / 1824 |
| 5 | -0.05 / -0.05 | -0.31 / -0.31 | 0.700 / 0.700 | 2085 / 2182 |
Keeping language: energy +0.00 (better on 0/10 readings), eat-fraction +0.000 (higher on 0/10)

## sparse2_language: information in the second symbol (bits above shuffled baseline, tick 100,000)
| seed | arm | P(signal) | food here: sym1 / sym2 adds | cluster direction: sym1 / sym2 adds |
|---|---|---|---|---|
| 1 | ON | 0.057 | +0.0113 / -0.0028 | +0.0149 / +0.0313 |
| 1 | OFF | 0.028 | +0.0057 / -0.0061 | +0.0829 / -0.0517 |
| 2 | ON | 0.016 | +0.0146 / +0.0032 | -0.0296 / -0.0146 |
| 2 | OFF | 0.019 | +0.0055 / -0.0331 | +0.0346 / +0.0796 |
| 3 | ON | 0.064 | -0.0026 / +0.0161 | +0.0247 / +0.0186 |
| 3 | OFF | 0.047 | +0.0326 / -0.0048 | -0.0202 / +0.0351 |
| 4 | ON | 0.022 | +0.0246 / +0.0058 | +0.0163 / +0.0242 |
| 4 | OFF | 0.034 | +0.0096 / -0.0053 | +0.0374 / +0.0519 |
| 5 | ON | 0.053 | +0.0034 / +0.0079 | -0.0125 / +0.0630 |
| 5 | OFF | 0.082 | -0.0049 / -0.0079 | +0.0065 / +0.0476 |
ON: second symbol adds +0.0245 bits about cluster direction on average
OFF: second symbol adds +0.0325 bits about cluster direction on average

## sparse2_cofood: shared food (worth 8, lasts 30 ticks, needs 2 eaters)
| seed | eaten share 0-10k -> 90-100k | expired | trend after removing population (r) | wasted solo tries per item per 1000 agents |
|---|---|---|---|---|
| 1 | 0.19 -> 0.44 | 0.80 -> 0.56 | +0.71 | 1.76 -> 1.28 |
| 2 | 0.20 -> 0.42 | 0.79 -> 0.57 | -0.00 | 1.63 -> 1.32 |
| 3 | 0.23 -> 0.53 | 0.76 -> 0.47 | +0.41 | 1.62 -> 1.88 |
| 4 | 0.20 -> 0.49 | 0.79 -> 0.51 | +0.24 | 1.98 -> 1.63 |
| 5 | 0.15 -> 0.60 | 0.84 -> 0.39 | +0.25 | 1.89 -> 1.42 |

## Novelty meter (all runs): behaviour change per 1,000 ticks (L1, 0-2), early vs late; lineages alive at the end
| experiment | arm | change, ticks 2-20k | change, 80-100k | lineages at end (5 seeds) |
|---|---|---|---|---|
| sparse2_memory | ON | 0.090 | 0.062 | 1, 1, 1, 1, 1 |
| sparse2_memory | OFF | 0.082 | 0.056 | 1, 1, 1, 1, 1 |
| sparse2_comm | ON | 0.082 | 0.056 | 1, 1, 1, 1, 1 |
| sparse2_comm | OFF | 0.073 | 0.063 | 1, 1, 1, 1, 1 |
| sparse2_language | ON | 0.085 | 0.066 | 1, 1, 1, 1, 1 |
| sparse2_language | OFF | 0.082 | 0.056 | 1, 1, 1, 1, 1 |
| sparse2_cofood | ON | 0.095 | 0.064 | 1, 1, 1, 1, 1 |
| sparse2_cofood | OFF | 0.076 | 0.070 | 1, 1, 1, 1, 1 |
