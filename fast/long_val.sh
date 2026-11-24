#!/bin/bash
cd /home/claude/lab/fast
export NUMBA_NUM_THREADS=1
for s in 2 3 4 5; do
  python3 run_fast.py val/long/cofood_on_seed$s --seed $s --ticks 25000 --cofood >> val/long_on_$s.txt 2>&1 &
  python3 run_fast.py val/long/cofood_off_seed$s --seed $s --ticks 25000 >> val/long_off_$s.txt 2>&1
  wait
done
echo ALLDONE > val/long_done.txt
