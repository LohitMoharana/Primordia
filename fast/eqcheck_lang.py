import sys, copy, numpy as np
sys.path.insert(0,'/home/claude/lab/stages/13_cofood')
import engine
base = engine.from_old_checkpoint('/home/claude/lab/fast/val/old_on_seed1/checkpoint_1000.pkl','/home/claude/lab/stages/13_cofood')
for lang, mem in [(False, False), (True, False), (True, True)]:
    a = copy.deepcopy(base); a.cfg['language'] = lang; a.cfg['memory'] = mem
    b = copy.deepcopy(a); a.use_numba = False; b.use_numba = True
    for t in range(200): a.step(); b.step()
    i = a.alive
    print(f"language={lang} memory={mem}: pop {a.population} {b.population} | energies match:",
          a.population == b.population and np.allclose(a.energy[a.alive], b.energy[b.alive]),
          "| second symbols match:", bool((a.last_sym2[a.alive] == b.last_sym2[b.alive]).all()) if a.population == b.population else None,
          "| share of signals with 2nd symbol:", round(float((b.last_sym2[b.alive] > 0).mean()), 3))
