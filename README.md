# Primordia

An artificial-life world where simple agents evolve with **nothing scripted**. Agents have small neural-network brains, sense their surroundings, eat, move, signal and reproduce with mutation. Whatever behaviour appears has to be discovered by evolution, not written in.

The question: what does a world need before agents evolve something genuinely new (cooperation, communication, memory, language) rather than the obvious "find food, eat, stay near others"?

## How experiments work

Every feature is tested as an **ablation**: two worlds with the same brain architecture and the same founders, differing in one feature only. Each pair is run on many random seeds (usually 10).

Verdicts come from **lesion tests**, not survival alone: take the evolved agents, give them equal energy and age, and compare them with and without the feature's information. Early single-seed "wins" turned out to be noise, so a result counts only if it holds across seeds.

## Results so far

| Feature | Verdict |
|---|---|
| Social sensing (seeing crowding nearby) | **Real and used.** Removing it cuts eating on 10/10 seeds. |
| Communication (4 symbols) | Not used. |
| Seasons | No better brains (an earlier "advantage" was an artefact of test timing). |
| Hazards (poison cells) | **Poison avoidance is real.** |
| Working memory | Does not help, even when memory weights start at zero. |
| Shared food (needs two eaters) | No coordination; agents just crowd. |
| Two-symbol language | Not used in open worlds. **First hint in terrain** (walls and rooms): better survival on 5/5 seeds vs 2/5 and +0.15 energy in lesion tests, but signals carry little information. Being checked on more seeds. |

Also measured: a **novelty meter** (behaviour change per 1,000 ticks levels off at about 0.06) and **family trees** (every run ends descended from one of 300 founders).

Full write-up, with every table and correction: [`FINDINGS.md`](FINDINGS.md).

## Code

| Folder | What it is |
|---|---|
| `fast/` | The current engine. `engine.py` (NumPy + Numba, ~110-180x the original), `engine_torch.py` (PyTorch/GPU), `kernels.py`, run and batch scripts, analysis and lesion tools. |
| `stages/` | The original pure-Python engine, one folder per roadmap stage (01 social sensing to 13 shared food). Includes the 300 evolved stage-5 founders (`stages/05_hazards/runs/hazards_on_seed1/checkpoint_23500.pkl`) that later experiments start from. |
| `lab/` | Tools for the original engine: survival evaluation, replication, analyses. |
| `docs/notes/` | Older handoff notes and per-experiment write-ups. |

Run outputs (logs and checkpoints, tens of GB) are not in the repo.

## Quick start

```bash
pip install numpy numba          # torch optional, for the GPU engine
# one run: terrain world with two-symbol language, 100,000 ticks
python fast/run_fast.py fast/runs/test --mode terrain_language --arm on --seed 1 --ticks 100000
# a batch: both arms, 5 seeds, 8 runs at a time (resumable)
python fast/replicate_modes.py --modes terrain_language --seeds 1 2 3 4 5 --jobs 8
# lesion test on a saved population
python -c "import sys; sys.path.insert(0,'fast'); import evaluate_fast as v; p='fast/runs/test/checkpoint_100000.pkl'; print(v.profile(p, reset_state=True), v.profile(p, reset_state=True, override={'language': False}))"
```

Modes (see `fast/modes.py`): `stage1`, `stage2`, `stage3`, `stage5`, `stage13`, `sparse2_memory`, `sparse2_memory0`, `sparse2_comm`, `sparse2_language`, `sparse2_cofood`, `terrain_language`.

Speed on a laptop (16 threads): about 160 ticks/s with ~10,400 agents on the CPU engine.

## Roadmap

1. ~~Fast engine and GPU version~~ (done)
2. ~~Sparse world: drifting food patches, terrain~~ (done)
3. Long runs (1,000,000 ticks) with novelty meter and family trees (running)
4. Worlds that push for novelty:
   - an **arms race** with predators that evolve too
   - **several ways to live** (food types needing different skills)
   - agents that **change the world** (storing food, building, lasting marks)
5. Bigger brains, only where needed
6. Many-seed sweeps
7. 3D, much later

## License

MIT, see [`LICENSE`](LICENSE).
