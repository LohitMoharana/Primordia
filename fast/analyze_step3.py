"""Step 3 summary: terrain runs extended to 1,000,000 ticks. Uses novelty.csv and log.csv only (fast)."""
import csv, glob, os, numpy as np
R = "runs/35_terrain_language"
acts = ["stay","N","S","E","W","eat","reproduce","signal_1","signal_2","signal_3","signal_4","share","mate","mark"]
rows = []
for d in sorted(glob.glob(f"{R}/language_*")):
    if os.path.exists(f"{d}/EXTINCT"): continue
    nv = list(csv.DictReader(open(f"{d}/novelty.csv")))
    T = np.array([float(r["tick"]) for r in nv]); ch = np.array([float(r["change_from_previous_window"] or "nan") for r in nv])
    def win(a, b): m = (T > a) & (T <= b); return np.nanmean(ch[m])
    def shares(t): r = min(nv, key=lambda r: abs(float(r["tick"]) - t)); return np.array([float(r[a]) for a in acts])
    lg = np.genfromtxt(f"{d}/log.csv", delimiter=",", names=True)
    pop = lambda a, b: lg["population"][(lg["tick"] > a) & (lg["tick"] <= b)].mean()
    en = lambda a, b: lg["mean_energy"][(lg["tick"] > a) & (lg["tick"] <= b)].mean()
    s1, s2 = shares(100000), shares(1000000)
    rows.append((os.path.basename(d), win(0, 10000), win(90000, 100000), win(490000, 500000), win(900000, 1000000),
                 int(nv[-1]["lineages_alive"]), pop(90000, 100000), pop(900000, 1000000), en(90000, 100000), en(900000, 1000000),
                 np.abs(s2 - s1).sum(), s1, s2, float(nv[-1]["weight_spread"]), float(nv[9]["weight_spread"]) if len(nv) > 9 else np.nan, T[-1]))
print("run | change 0-10k | 90-100k | 490-500k | 900k-1M | lineages | pop 100k | pop 1M | energy 100k | energy 1M | behaviour shift 100k->1M | last tick")
for r in rows:
    print(f"{r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]} | {r[6]:.0f} | {r[7]:.0f} | {r[8]:.2f} | {r[9]:.2f} | {r[10]:.3f} | {r[15]:.0f}")
A = np.array([r[1:5] for r in rows]); print("mean change by window:", np.round(np.nanmean(A, 0), 3))
s1 = np.mean([r[11] for r in rows], 0); s2 = np.mean([r[12] for r in rows], 0)
print("action shares 100k -> 1M:", ", ".join(f"{a} {x:.3f}->{y:.3f}" for a, x, y in zip(acts, s1, s2)))
print("pop 100k->1M:", np.mean([r[6] for r in rows]).round(), "->", np.mean([r[7] for r in rows]).round(),
      "| energy:", np.mean([r[8] for r in rows]).round(2), "->", np.mean([r[9] for r in rows]).round(2))
print("weight spread at 10k -> 1M:", np.nanmean([r[14] for r in rows]).round(3), "->", np.mean([r[13] for r in rows]).round(3))
