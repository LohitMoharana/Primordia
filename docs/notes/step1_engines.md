
## Step 1 complete: fast engines (24 Nov 2026)
* fast/engine.py (NumPy + Numba): ~110x the old engine at ~1,500 agents on 2 cores (6.1 -> 674 ticks/s,
  same machine and state); exact match with its own NumPy path; statistical match with the old engine
  (stage 13 over 15,000 ticks; stages 1, 2, 3, 5 population histories on seeds 1-5).
* fast/engine_torch.py (PyTorch, GPU): same rules, same checkpoint format; matches the numba engine
  statistically (stage 13 and sparse worlds, 5 seeds each).
* Laptop benchmark (128 x 128 world, ~10,400 agents; bench_gpu.py):
  numba CPU, 16 threads: 160 ticks/s; torch on RTX 5060 Laptop GPU: 110 ticks/s; old engine
  (estimated from its 6.1 ticks/s at 1,500 agents): ~0.9 ticks/s. So ~180x on the CPU engine.
  The GPU engine is slower here because each tick still syncs with the CPU several times (sharing,
  shared food, bookkeeping); it should overtake at much larger populations or once those steps move
  to the GPU. For now the numba engine is the one to use.
* Added for steps 2-3: working memory, two-symbol language, sparse world v1 and v2 (vanishing
  clusters), novelty meter (novelty.csv) and family trees (births.csv), both engines.
