#!/bin/bash
# shared food ON/OFF, seeds 6-10, fast engine, 2 at a time (this machine has 2 cores)
cd /home/claude/lab/fast
export NUMBA_NUM_THREADS=1
for s in 6 7 8 9 10; do
  python3 run_fast.py val/long/cofood_on_seed$s --seed $s --ticks 25000 --cofood > val/long_on_$s.txt 2>&1 &
  python3 run_fast.py val/long/cofood_off_seed$s --seed $s --ticks 25000 > val/long_off_$s.txt 2>&1
  wait
done
echo done > val/auto_cofood_done.txt
