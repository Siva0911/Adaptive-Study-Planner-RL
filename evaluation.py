"""Evaluation on continuous trajectories and sensible observable-only baselines."""

import math
import numpy as np
from study_env import StudyEnv, ACTIONS

FIXED_CYCLES = (
    (0, 0, 1, 2, 3),
    (0, 0, 1, 3),
    (0, 1, 3),
    (0, 0, 3),
    (0, 3, 1, 3),
    (0, 1, 3, 3),
    (0, 3),
    (0, 3, 0, 1, 3, 3),
    (0, 0, 3, 1, 3),
    (0, 1, 2, 3, 3),
)
HEURISTICS = []
for bins in (3, 5):
    for fatigue_bin in (1, 2):
        for review_bin in (1, 2):
            for late_review in (False, True):
                HEURISTICS.append(
                    dict(
                        bins=bins,
                        fatigue_bin=fatigue_bin,
                        review_bin=review_bin,
                        late_review=late_review,
                    )
                )
HEURISTICS = tuple(HEURISTICS)


# Choose a sensible activity using observable threshold rules.
# Check fatigue first, then decide between review, practice, and study.
def heuristic_action(environment, settings):
    bins = settings["bins"]
    n, c, f = (min(bins - 1, int(x * bins)) for x in environment.state)
    if f >= settings["fatigue_bin"]:
        return 3
    if n >= settings["review_bin"] or (
        settings["late_review"] and environment.remaining <= 3 and (n > 0)
    ):
        return 1
    if c >= bins - 1 and environment.history == 0 and (environment.remaining > 3):
        return 2
    return 0


# Measure one policy on one student type without learning.
# Run reproducible episodes and summarize scores, burnout, and action counts.
def evaluate(config, policy, student, episodes, seed, reference, bins=3):
    environment = StudyEnv(config, bins, seed)
    action_random = np.random.default_rng(np.random.SeedSequence([seed, 991]))
    episode_results = []
    counts = np.zeros(4, dtype=int)
    for episode in range(episodes):
        observation = environment.reset(student)
        task_return = objective_return = 0.0
        fatigue = []
        for step in range(config.horizon):
            kind = policy["kind"]
            if kind == "learner":
                action = int(policy["q"][observation].argmax())
            elif kind == "oracle":
                action = reference.act(environment.state, environment.remaining)
            elif kind == "fixed":
                cycle = policy["cycle"]
                action = cycle[step % len(cycle)]
            elif kind == "heuristic":
                action = heuristic_action(environment, policy["settings"])
            elif kind == "random":
                action = int(action_random.integers(4))
            else:
                raise ValueError("Unknown policy kind")
            counts[action] += 1
            observation, _, done, info = environment.step(action)
            task_return += config.gamma**step * info["task_reward"]
            objective_return += config.gamma**step * info["objective_reward"]
            fatigue.append(info["fatigue"])
            if done:
                break
        episode_results.append(
            (
                task_return,
                objective_return,
                info["score"],
                info["burnout"],
                environment.n,
                environment.c,
                np.mean(fatigue),
                max(fatigue),
                step + 1,
            )
        )
    values = np.asarray(episode_results, dtype=float)
    mean = values.mean(axis=0)
    reference_value = reference.initial_value
    result = dict(
        task_return=float(mean[0]),
        objective_return=float(mean[1]),
        exam_score=float(mean[2]),
        burnout_rate=float(mean[3]),
        final_new_knowledge=float(mean[4]),
        final_consolidated_knowledge=float(mean[5]),
        mean_fatigue=float(mean[6]),
        max_fatigue=float(mean[7]),
        mean_steps=float(mean[8]),
        numerical_reference_value=reference_value,
        reference_ratio=(
            float(mean[1] / reference_value) if reference_value > 1e-08 else None
        ),
        reference_gap=float(reference_value - mean[1]),
    )
    for name, count in zip(ACTIONS, counts):
        result["count_" + name.lower()] = int(count)
    return result


# Summarize evaluation metrics by population and method.
# Average student types within each seed and compute intervals across seeds.
def aggregate(rows):
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.16,
        14: 2.145,
        15: 2.131,
        16: 2.12,
        17: 2.11,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.08,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.06,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
    }
    result = []
    keys = (
        "task_return",
        "objective_return",
        "exam_score",
        "burnout_rate",
        "reference_ratio",
        "reference_gap",
        "final_new_knowledge",
        "final_consolidated_knowledge",
        "mean_fatigue",
        "max_fatigue",
        "mean_steps",
    )
    for population in sorted({row["population"] for row in rows}):
        for method in sorted({row["method"] for row in rows}):
            group = [
                row
                for row in rows
                if row["population"] == population and row["method"] == method
            ]
            if not group:
                continue
            seeds = sorted({row["seed"] for row in group})
            output = dict(population=population, method=method, seeds=len(seeds))
            for key in keys:
                means = []
                for seed in seeds:
                    values = [row[key] for row in group if row["seed"] == seed]
                    if any((v is None for v in values)):
                        continue
                    means.append(float(np.mean(values)))
                valid = len(means) == len(seeds)
                if valid:
                    output[key] = float(np.mean(means))
                else:
                    output[key] = None
                if valid and len(means) > 1:
                    output[key + "_ci95"] = float(
                        critical.get(len(means) - 1, 1.96)
                        * np.std(means, ddof=1)
                        / math.sqrt(len(means))
                    )
                else:
                    output[key + "_ci95"] = None
            if output["reference_ratio"] is not None:
                output["mean_meets_85_percent_reference"] = (
                    output["reference_ratio"] >= 0.85
                )
            else:
                output["mean_meets_85_percent_reference"] = None
            result.append(output)
    return result
