"""
Deduplicate a log.csv that has overlapping tick ranges from a crash + resume
(common before checkpoints saved RNG state — see sim.py's from_checkpoint).
Keeps the LAST row written for each tick (the final authoritative run),
sorted by tick.

Usage:
    python3 clean_log.py runs/social1/log.csv
Writes runs/social1/log.csv.clean — inspect it, then rename over the original.
"""
import csv
import sys

path = sys.argv[1]
with open(path) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

by_tick = {}
for row in rows:
    by_tick[int(row[0])] = row  # later occurrence overwrites earlier — "last wins"

out_path = path + ".clean"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for tick in sorted(by_tick):
        writer.writerow(by_tick[tick])

print(f"{len(rows)} rows -> {len(by_tick)} unique ticks. Wrote {out_path}")
