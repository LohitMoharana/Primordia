import csv, glob, numpy as np
def load(d):
    return {int(float(r['tick'])): r for r in csv.DictReader(open(d + '/log.csv'))}
old = {d: load(d) for d in sorted(glob.glob('val/old_on_seed*'))}
fast = [load(d) for d in sorted(glob.glob('val/fast_on_seed*'))]
print(f"{'tick':>5} | {'fast pop mean±sd':>17} {'range':>11} | " + " ".join(f"{d[-5:]:>14}" for d in old))
for t in range(250, 3001, 250):
    fp = np.array([float(f[t]['population']) for f in fast if t in f])
    fe = np.array([float(f[t]['mean_energy']) for f in fast if t in f])
    fg = np.array([float(f[t]['max_generation']) for f in fast if t in f])
    cells = []
    for d, o in old.items():
        cells.append(f"{int(float(o[t]['population'])):>5} e{float(o[t]['mean_energy']):.2f} g{o[t]['max_generation']:>2}" if t in o else " " * 14)
    print(f"{t:>5} | {fp.mean():7.0f} ± {fp.std():4.0f} e{fe.mean():.2f} g{fg.mean():4.1f} {int(fp.min()):>4}-{int(fp.max()):<4} | " + " ".join(cells))
for d in old:
    rows = list(csv.DictReader(open(d + '/cofood.csv')))
    r = {int(x['tick']): x for x in rows}
    if 3000 in r: print(d, {k: r[3000][k] for k in ['spawned', 'eaten', 'expired', 'solo_attempts', 'co_eaters']})
