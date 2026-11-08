"""Create the two report figures from a completed results folder.

Usage: python make_figures.py new_results images
Writes learning.pdf and comparison.pdf (plus PNG previews) into the output folder.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T975 = {2: 12.706, 3: 4.303, 5: 2.776, 10: 2.262}
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
})


# Mean and Student-t 95% half-width across seeds for one array of seed values.
def mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return values.mean(), 0.0
    return values.mean(), T975.get(n, 1.96) * values.std(ddof=1) / np.sqrt(n)


def learning_figure(root, out):
    rows = list(csv.DictReader(open(root / "learning_curves.csv")))
    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        episode = int(row["episode"])
        data[row["method"]][episode].append(
            (float(row["objective_return"]), 100 * float(row["burnout_rate"]))
        )
    styles = {
        "qlearning": ("Q-learning, 5 bins", BLUE, "-"),
        "sarsa": ("SARSA, 5 bins", ORANGE, "-"),
        "qlearning_matched": ("Q-learning, 3 bins", BLUE, "--"),
        "sarsa_matched": ("SARSA, 3 bins", ORANGE, "--"),
    }
    figure, axes = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
    labels = ["Training objective return", "Training burnout (%)"]
    for index, axis in enumerate(axes):
        for method, (label, color, line) in styles.items():
            episodes = sorted(data[method])
            thousands = [e / 1000 for e in episodes]
            stats = [mean_ci([v[index] for v in data[method][e]]) for e in episodes]
            mean = np.array([s[0] for s in stats])
            half = np.array([s[1] for s in stats])
            axis.plot(thousands, mean, color=color, linestyle=line, linewidth=1.4, label=label)
            axis.fill_between(thousands, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
        axis.set_ylabel(labels[index])
        axis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[1].set_xlabel(r"Training episodes ($\times 10^3$)")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].legend(loc="upper right", frameon=False)
    figure.tight_layout()
    figure.savefig(out / "learning.pdf")
    figure.savefig(out / "learning.png", dpi=200)
    plt.close(figure)


def comparison_figure(root, out):
    rows = list(csv.DictReader(open(root / "summary.csv")))
    table = {(r["population"], r["method"]): r for r in rows}
    methods = [
        ("fixed", "Fixed"), ("heuristic", "Heur."), ("qlearning", "Q5"),
        ("sarsa", "S5"), ("qlearning_matched", "Q3"), ("sarsa_matched", "S3"),
        ("oracle", "Ref."),
    ]
    x = np.arange(len(methods))
    width = 0.38
    figure, axis = plt.subplots(figsize=(3.5, 2.4))
    for offset, (population, label, color) in zip(
        (-width / 2, width / 2),
        (("training", "Training types", BLUE), ("test", "Test types", ORANGE)),
    ):
        mean = [100 * float(table[(population, m)]["reference_ratio"]) for m, _ in methods]
        half = [100 * float(table[(population, m)]["reference_ratio_ci95"]) for m, _ in methods]
        axis.bar(x + offset, mean, width - 0.04, color=color, label=label,
                 yerr=half, capsize=2, error_kw=dict(elinewidth=0.8, ecolor=INK))
    axis.axhline(85, color=MUTED, linestyle=":", linewidth=1, label="85% target")
    axis.set_xticks(x, [name for _, name in methods])
    axis.set_ylabel("Reference-normalized return (%)")
    axis.set_xlabel("Policy")
    axis.set_ylim(0, 120)
    axis.set_yticks(range(0, 101, 20))
    axis.grid(True, axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", ncol=3, frameon=False, handlelength=1.5, columnspacing=1.0)
    figure.tight_layout()
    figure.savefig(out / "comparison.pdf")
    figure.savefig(out / "comparison.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "new_results")
    output = Path(sys.argv[2] if len(sys.argv) > 2 else "images")
    output.mkdir(parents=True, exist_ok=True)
    learning_figure(results, output)
    comparison_figure(results, output)
    print(f"Wrote figures to {output}")
