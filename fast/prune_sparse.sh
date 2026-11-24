#!/bin/bash
# keep sparse-world runs small while they run: every 2 minutes, drop saves other than the readings and the latest
cd /home/claude/lab/fast
while pgrep -f replicate_modes.py > /dev/null; do
python3 - <<'PY'
import glob, os, re
for d in glob.glob('runs/2*_sparse_*/*'):
    c = {int(re.search(r'checkpoint_(\d+)', f).group(1)): f for f in glob.glob(os.path.join(d, 'checkpoint_*.pkl'))}
    if not c: continue
    for t, f in c.items():
        if t not in (22500, 25000, max(c)): os.remove(f)
PY
sleep 120; done
