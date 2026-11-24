"""
Plot a run's log.csv so you can actually see what happened instead of
reading raw numbers. Usage:

    python3 plot.py runs/seed0
    python3 plot.py runs/seed0 runs/seed1 runs/seed2   # overlay multiple seeds
"""
import sys
import csv
import matplotlib.pyplot as plt


def load_log(path):
    rows = {"tick": [], "population": [], "mean_energy": [], "mean_age": [],
            "max_generation": [], "mean_complexity": [], "total_food": []}
    with open(f"{path}/log.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in rows:
                rows[k].append(float(r[k]))
    return rows


def main(paths):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Artificial life — stage 1 run(s)")

    panels = [
        ("population", "Population"),
        ("mean_energy", "Mean energy"),
        ("mean_age", "Mean age (ticks)"),
        ("max_generation", "Deepest generation reached"),
        ("mean_complexity", "Mean |weight| (brain drift from random init)"),
        ("total_food", "Total food in world"),
    ]

    for ax, (key, title) in zip(axes.flat, panels):
        for path in paths:
            data = load_log(path)
            label = path.rstrip("/").split("/")[-1]
            ax.plot(data["tick"], data[key], label=label, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("tick")
        if len(paths) > 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = "run_summary.png"
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    paths = sys.argv[1:] or ["runs/seed0"]
    main(paths)
