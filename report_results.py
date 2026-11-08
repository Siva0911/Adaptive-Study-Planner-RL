"""Generate honest tables and plots from a specified run directory."""

from pathlib import Path
import argparse
import csv
import json


# Read CSV data into a list of rows.
# Open the requested file and keep its column names with each row.
def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# Build a readable findings report from one completed result folder.
# Read saved tables, explain the reference limits, and create the charts.
def generate(root):
    root = Path(root)
    summary = read(root / "summary.csv")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    lines = [
        "# Continuous Study Planner: measured results",
        "",
        "Generated from this run directory. Knowledge and fatigue evolve continuously; only learner observations are discretized.",
        "",
        f"Run status: {protocol['status']}. Python {protocol['python']}; NumPy {protocol['numpy']}.",
        "",
        "## Main evaluation",
        "",
        "| Population | Method | Exam score (0-1) | Burnout % | Numerical-reference return % | Approx. 95% interval % |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        ratio = row["reference_ratio"]
        interval = row["reference_ratio_ci95"]
        if ratio:
            value = f"{100 * float(ratio):.2f}"
        else:
            value = "unavailable"
        if ratio and interval:
            interval = f"{100 * (float(ratio) - float(interval)):.2f} to {100 * (float(ratio) + float(interval)):.2f}"
        else:
            interval = "unavailable"
        lines.append(
            f"| {row['population']} | {row['method']} | {float(row['exam_score']):.4f} | {100 * float(row['burnout_rate']):.2f} | {value} | {interval} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Reference ratios compare learner objective return with a known-type numerical planning estimate. The reference uses interpolation and Gaussian quadrature and is NOT a proven upper bound or an exact continuous-state optimum.",
        "- An 85% ratio is a research target, not classification accuracy, a guaranteed exam score, or a submission guarantee. Negative ratios are possible; ratios are unavailable when the reference value is nonpositive.",
        "- Raw task return awards final exam score at normal completion and a burnout penalty otherwise. Objective return additionally subtracts the configured fatigue penalty. Planning and evaluation use the same objective.",
        "- Approximate 95% Student-t intervals use independent seed means (normal approximation above 30 seeds). One-seed runs do not have seed-level intervals. Type results and action counts are in metrics.csv.",
        "- Test students are excluded from validation selection. Pilot scenarios use training/validation students only. Comparisons between separately tuned learners do not isolate on-policy versus off-policy effects; use the optional matched-settings methods and training curves for that question.",
        "- Burnout scores are zero, and burnout episodes remain in every average. Zero observed burnout is not evidence of zero true risk.",
        "",
        "## Numerical-reference checks",
        "",
        "| Validation student (mu,phi) | Grid | Quadrature points | Predicted value | Evaluated policy return | Burnout % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in read(root / "oracle_refinement.csv"):
        lines.append(
            f"| ({row['mu']}, {row['phi']}) | {row['grid']} | {row['quadrature']} | {float(row['predicted_reference']):.4f} | {float(row['evaluated_return']):.4f} | {100 * float(row['burnout_rate']):.2f} |"
        )
    lines += [
        "",
        "Inspect refinement differences before treating the reference ratio as precise. More mesh points improve resolution but do not remove all approximation error. The evaluated reference policy may differ from its interpolated predicted value.",
        "",
        "## Sensitivity pilots",
        "",
        "These runs transfer selected learner settings to new scenarios and use fewer seeds/episodes than the main run. They provide preliminary evidence rather than a fully powered comparison.",
        "",
        "| Scenario | Population | Learner | Exam score | Burnout % | Reference ratio % |",
        "|---|---|---|---:|---:|---:|",
    ]
    for path in sorted((root / "scenarios").glob("*/summary.csv")):
        for row in read(path):
            if row["method"] not in ("qlearning", "sarsa"):
                continue
            if row["reference_ratio"]:
                ratio = f"{100 * float(row['reference_ratio']):.2f}"
            else:
                ratio = "unavailable"
            lines.append(
                f"| {path.parent.name} | {row['population']} | {row['method']} | {float(row['exam_score']):.4f} | {100 * float(row['burnout_rate']):.2f} | {ratio} |"
            )
    if not (root / "scenarios").exists():
        lines += ["", "No sensitivity pilots were executed in this run."]
    lines += [
        "",
        "## Scope",
        "",
        "This is a hand-designed student simulator. Continuous forgetting, Gaussian study noise, consolidation, fatigue, burnout, finite time, and hidden student types are implemented. Recent practice performance summarizes the latest three attempts. A small tabular observation does not make the hidden process fully observable. Real-student efficacy and DQN are outside this implementation.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    plot(root, summary)


# Draw performance and training charts from saved result tables.
# Compare methods and show learning progress and burnout over time.
def plot(root, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    populations = list(dict.fromkeys((row["population"] for row in summary)))
    figure, axes = plt.subplots(
        len(populations), 3, figsize=(15, 4 * len(populations)), squeeze=False
    )
    for i, population in enumerate(populations):
        rows = [row for row in summary if row["population"] == population]
        for j, key in enumerate(("exam_score", "burnout_rate", "reference_ratio")):
            valid = [row for row in rows if row[key] != ""]
            axis = axes[i, j]
            values = np.array([float(row[key]) for row in valid])
            errors = np.array([float(row[key + "_ci95"] or 0.0) for row in valid])
            if key in ("exam_score", "burnout_rate"):
                lower = np.minimum(errors, np.maximum(0.0, values))
            else:
                lower = errors
            if key in ("exam_score", "burnout_rate"):
                upper = np.minimum(errors, np.maximum(0.0, 1 - values))
            else:
                upper = errors
            axis.bar(
                [row["method"] for row in valid],
                values,
                yerr=np.array([lower, upper]),
                capsize=3,
            )
            axis.set_title(population + ": " + key)
            axis.tick_params(axis="x", rotation=45)
            if key == "reference_ratio":
                axis.axhline(
                    0.85, color="red", linestyle="--", label="85% research target"
                )
                axis.legend()
    figure.tight_layout()
    figure.savefig(root / "comparison.png", dpi=150)
    plt.close(figure)
    curves = read(root / "learning_curves.csv")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in sorted({row["method"] for row in curves}):
        episodes = sorted(
            {int(row["episode"]) for row in curves if row["method"] == method}
        )
        for axis, key in zip(axes, ("objective_return", "burnout_rate")):
            values = [
                np.mean(
                    [
                        float(row[key])
                        for row in curves
                        if row["method"] == method and int(row["episode"]) == ep
                    ]
                )
                for ep in episodes
            ]
            axis.plot(episodes, values, label=method)
            axis.set(xlabel="Training episodes", ylabel=key)
            axis.legend()
    figure.tight_layout()
    figure.savefig(root / "learning.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results")
    generate(Path(parser.parse_args().directory))
