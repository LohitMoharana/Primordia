#!/bin/bash
cd /home/claude/lab/fast
python3 replicate_modes.py --modes sparse_memory sparse_comm sparse_cofood --seeds 1 2 3 4 5 --jobs 2 > runs_sparse.log 2>&1
