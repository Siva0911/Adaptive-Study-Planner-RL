"""Verify a chosen run directory, including recomputation of its summary."""

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from evaluation import aggregate


# Read CSV data into a list of rows.
# Open the requested file and keep its column names with each row.
def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# Check that saved experiments are complete and internally consistent.
# Recompute summaries and verify metrics, model shapes, and training counts.
def check(root):
    run_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    metrics = read(root / "metrics.csv")
    curves = read(root / "learning_curves.csv")
    methods = ("random", "fixed", "heuristic", "oracle", "qlearning", "sarsa") + (
        ("qlearning_matched", "sarsa_matched")
        if run_config["matched_comparison"]
        else ()
    )
    expected = (
        run_config["seeds"]
        * len(methods)
        * sum((len(v) for v in run_config["populations"].values()))
    )
    assert len(metrics) == expected, f"{root}: wrong metric row count"
    keys = [
        (row["seed"], row["population"], row["method"], row["mu"], row["phi"])
        for row in metrics
    ]
    assert len(keys) == len(set(keys)), "Duplicate metric rows"
    numeric = []
    for saved_metrics in metrics:
        row = {}
        for key, value in saved_metrics.items():
            if key in ("population", "method"):
                row[key] = value
            elif key == "seed":
                row[key] = int(value)
            elif value == "":
                row[key] = None
            else:
                row[key] = float(value)
                assert math.isfinite(row[key]), f"Nonfinite {key}"
        assert 0 <= row["exam_score"] <= 1
        assert 0 <= row["burnout_rate"] <= 1
        assert (
            row["final_new_knowledge"] + row["final_consolidated_knowledge"]
            <= 1 + 1e-10
        )
        assert (
            abs(
                row["reference_gap"]
                - (row["numerical_reference_value"] - row["objective_return"])
            )
            < 1e-10
        )
        if (
            run_config["config"]["gamma"] == 1
            and run_config["config"]["fatigue_weight"] == 0
        ):
            assert (
                abs(
                    row["task_return"]
                    - (
                        row["exam_score"]
                        - run_config["config"]["burnout_penalty"] * row["burnout_rate"]
                    )
                )
                < 1e-10
            )
        numeric.append(row)
    regenerated = aggregate(numeric)
    saved = read(root / "summary.csv")
    assert len(regenerated) == len(saved)
    for expected_row, saved_row in zip(regenerated, saved):
        for key, value in expected_row.items():
            if value is None:
                assert saved_row[key] == ""
            elif isinstance(value, float):
                assert abs(value - float(saved_row[key])) < 1e-10, (
                    key,
                    value,
                    saved_row[key],
                )
            else:
                assert str(value) == saved_row[key]
    seeds = sorted({int(row["seed"]) for row in metrics})
    assert len(seeds) == run_config["seeds"]
    trained = [m for m in methods if m.startswith(("qlearning", "sarsa"))]
    for seed in seeds:
        for method in trained:
            saved = np.load(root / f"{method}_seed{seed}.npz", allow_pickle=False)
            learner = json.loads(str(saved["learner_json"]))
            expected_shape = (
                learner["bins"] ** 3 * (run_config["config"]["horizon"] + 1) * 3,
                4,
            )
            assert saved["q"].shape == expected_shape
            assert np.isfinite(saved["q"]).all()
            assert json.loads(str(saved["config_json"])) == run_config["config"]
            logs = [
                row
                for row in curves
                if int(row["seed"]) == seed and row["method"] == method
            ]
            assert (
                sum((int(row["block_episodes"]) for row in logs))
                == run_config["episodes"]
            )
            assert max((int(row["episode"]) for row in logs)) == run_config["episodes"]
    print(
        f"PASS {root.name}: {len(metrics)} evaluation rows, {len(seeds) * len(trained)} complete training runs; summary matches"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", nargs="?", default="results")
    root = Path(p.parse_args().directory)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol["status"] != "complete":
        raise RuntimeError("Run has not completed successfully")
    check(root)
    for child in sorted((root / "scenarios").glob("*/config.json")):
        check(child.parent)
    print("All checks passed.")
