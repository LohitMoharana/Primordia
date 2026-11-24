import sys, json, os, red_queen as q
out = "runs/40_arms_race/red_queen.jsonl"
for name in sys.argv[1:]:
    d = f"runs/40_arms_race/{name}"
    t = q.table(d, 50000, 1000000, reps=3)
    with open(out, "a") as f:
        f.write(json.dumps({"run": name, **{f"prey{a}_pred{b}": v for (a, b), v in t.items()}}) + "\n")
    print(name, {f"{a}/{b}": round(v, 2) for (a, b), v in t.items()}, flush=True)
