"""Run a complete reproducible continuous-state study-planning experiment.

python run_project.py --profile smoke
python run_project.py --profile full --output results
"""

from dataclasses import asdict, replace, fields
from operator import itemgetter
import configparser
from pathlib import Path
import argparse
import csv
import json
import platform
import time
import numpy as np
from study_env import Config, TRAIN_TYPES, VALIDATION_TYPES, TEST_TYPES
from agents import LearnerConfig, train
from oracle import NumericalOracle
from evaluation import FIXED_CYCLES, HEURISTICS, evaluate, aggregate

PROFILES = (
    LearnerConfig(),
    LearnerConfig(alpha=0.15, epsilon_final=0.0),
    LearnerConfig(bins=5),
    LearnerConfig(bins=5, alpha=0.15, epsilon_final=0.0),
)
SCENARIOS = {
    "short_h8": dict(horizon=8),
    "medium_h14": dict(horizon=14),
    "sparse": dict(shaped=False),
    "fatigue_penalty": dict(fatigue_weight=0.01),
    "burnout_penalty_2": dict(burnout_penalty=2.0),
    "no_noise": dict(noise_std=0.0),
    "high_noise": dict(noise_std=0.06),
    "discount_099": dict(gamma=0.99),
}


# Save experiment rows with one header and consistent columns.
# Write nothing when there are no rows to save.
def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Save run settings as readable JSON.
# Reject invalid numerical values instead of silently recording them.
def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


# Compare fixed schedules and threshold rules on validation students.
# Return the best fixed schedule, best heuristic, and all candidate scores.
def select_baselines(cfg, evaluation, grid):
    records = []
    candidates = [dict(kind="fixed", cycle=p) for p in FIXED_CYCLES] + [
        dict(kind="heuristic", settings=p) for p in HEURISTICS
    ]
    references = [NumericalOracle(cfg, s, grid) for s in VALIDATION_TYPES]
    for index, policy in enumerate(candidates):
        scores = []
        for ti, (student, reference) in enumerate(zip(VALIDATION_TYPES, references)):
            row = evaluate(cfg, policy, student, evaluation, 470000 + ti, reference)
            scores.append(row["objective_return"])
        records.append(
            dict(
                candidate=index,
                kind=policy["kind"],
                settings=json.dumps(policy),
                validation_objective=float(np.mean(scores)),
            )
        )
    fixed = 0
    heuristic = len(FIXED_CYCLES)
    for index, row in enumerate(records):
        if index < len(FIXED_CYCLES):
            if row["validation_objective"] > records[fixed]["validation_objective"]:
                fixed = index
        elif row["validation_objective"] > records[heuristic]["validation_objective"]:
            heuristic = index
    return (candidates[fixed], candidates[heuristic], records)


# Save completed metrics, learning curves, and configuration together.
# Sort rows consistently and calculate the population summaries.
def save_results(
    directory,
    outputs,
    cfg,
    selected,
    baseline,
    episodes,
    evaluation,
    grid,
    populations,
    matched,
):
    rows = sorted(
        [r for output in outputs for r in output[0]],
        key=itemgetter("seed", "population", "method", "mu", "phi"),
    )
    curves = sorted(
        [r for output in outputs for r in output[1]],
        key=itemgetter("seed", "method", "episode"),
    )
    write_csv(directory / "metrics.csv", rows)
    write_csv(directory / "learning_curves.csv", curves)
    write_csv(directory / "summary.csv", aggregate(rows))
    write_csv(directory / "baseline_validation.csv", baseline[2])
    write_json(
        directory / "config.json",
        dict(
            config=asdict(cfg),
            selected_learners=selected,
            fixed=baseline[0],
            heuristic=baseline[1],
            episodes=episodes,
            eval_episodes=evaluation,
            seeds=len(outputs),
            oracle_grid=grid,
            quadrature=7,
            populations=populations,
            matched_comparison=matched,
            timing=sorted([o[2] for o in outputs], key=itemgetter("seed")),
        ),
    )


# Check the sensitivity of the approximate planning reference.
# Compare larger meshes and more Gaussian integration points on validation students.
def oracle_refinement(cfg, grid, evaluation):
    records = []
    for ti, student in enumerate(VALIDATION_TYPES):
        for mesh in (grid, grid + 8):
            oracle = NumericalOracle(cfg, student, mesh, quadrature=7)
            row = evaluate(
                cfg, dict(kind="oracle"), student, evaluation, 560000 + ti, oracle
            )
            records.append(
                dict(
                    mu=student[0],
                    phi=student[1],
                    grid=mesh,
                    quadrature=7,
                    predicted_reference=oracle.initial_value,
                    evaluated_return=row["objective_return"],
                    burnout_rate=row["burnout_rate"],
                )
            )
        oracle = NumericalOracle(cfg, student, grid, quadrature=11)
        row = evaluate(
            cfg, dict(kind="oracle"), student, evaluation, 560000 + ti, oracle
        )
        records.append(
            dict(
                mu=student[0],
                phi=student[1],
                grid=grid,
                quadrature=11,
                predicted_reference=oracle.initial_value,
                evaluated_return=row["objective_return"],
                burnout_rate=row["burnout_rate"],
            )
        )
    return records


# Read project.config and apply explicit command-line overrides.
# Return the selected run settings and validated environment configuration.
def read_settings(args):
    settings = configparser.ConfigParser()
    path = Path(args.config)
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    settings.read(path, encoding="utf-8")
    if args.profile is None:
        args.profile = settings.get("run", "profile")
    if args.profile not in ("full", "smoke"):
        raise ValueError("profile must be full or smoke")
    if args.output is None:
        args.output = settings.get("run", "output")
    for name in ("matched", "skip_scenarios"):
        if getattr(args, name) is None:
            setattr(args, name, settings.getboolean("run", name))
    for name in (
        "episodes",
        "seeds",
        "eval_episodes",
        "tuning_episodes",
        "scenario_episodes",
        "scenario_seeds",
        "oracle_grid",
    ):
        if getattr(args, name) is None:
            setattr(args, name, settings.getint(args.profile, name))
    model = {}
    for field in fields(Config):
        if isinstance(field.default, bool):
            model[field.name] = settings.getboolean("environment", field.name)
        elif isinstance(field.default, int):
            model[field.name] = settings.getint("environment", field.name)
        else:
            model[field.name] = settings.getfloat("environment", field.name)
    return (args, Config(**model))


# Train each learner and evaluate the policies for one seed.
# Save its Q-tables and return its student metrics and learning curves.
def run_seed(
    config,
    selected,
    baseline,
    seed,
    episodes,
    evaluation,
    grid,
    directory,
    populations,
    matched,
):
    cfg = config
    out = directory
    policies = dict(
        random=dict(kind="random"),
        fixed=baseline[0],
        heuristic=baseline[1],
        oracle=dict(kind="oracle"),
    )
    learner_configs = {}
    curves = []
    elapsed = {}
    for method in ("qlearning", "sarsa"):
        learner = LearnerConfig(**selected[method])
        before = time.perf_counter()
        q, learning = train(cfg, learner, method, seed, episodes)
        elapsed[method] = time.perf_counter() - before
        np.savez_compressed(
            out / f"{method}_seed{seed}.npz",
            q=q,
            learner_json=json.dumps(asdict(learner)),
            config_json=json.dumps(asdict(cfg)),
        )
        policies[method] = dict(kind="learner", q=q)
        learner_configs[method] = learner
        curves.extend((dict(seed=seed, method=method, **r) for r in learning))
    if matched:
        common = LearnerConfig()
        for method in ("qlearning", "sarsa"):
            q, learning = train(cfg, common, method, seed + 3000, episodes)
            name = method + "_matched"
            policies[name] = dict(kind="learner", q=q)
            learner_configs[name] = common
            np.savez_compressed(
                out / f"{name}_seed{seed}.npz",
                q=q,
                learner_json=json.dumps(asdict(common)),
                config_json=json.dumps(asdict(cfg)),
            )
            curves.extend((dict(seed=seed, method=name, **r) for r in learning))
    rows = []
    for pi, (population, students) in enumerate(populations.items()):
        for ti, student in enumerate(students):
            reference = NumericalOracle(cfg, student, grid)
            for method, policy in policies.items():
                if method in learner_configs:
                    bins = learner_configs[method].bins
                else:
                    bins = 3
                metrics = evaluate(
                    cfg,
                    policy,
                    student,
                    evaluation,
                    1900000 + seed * 10000 + pi * 1000 + ti,
                    reference,
                    bins,
                )
                rows.append(
                    dict(
                        seed=seed,
                        population=population,
                        method=method,
                        mu=student[0],
                        phi=student[1],
                        **metrics,
                    )
                )
    return (rows, curves, dict(seed=seed, **elapsed))


# Train each candidate on training students and score it on validation students.
# Freeze the best settings for each algorithm before evaluating any test student.
def select_learners(config, settings, directory):
    tuning = []
    tuning_seeds = (101,) if settings.profile == "smoke" else (101, 102)
    for method in ("qlearning", "sarsa"):
        for index, learner in enumerate(PROFILES):
            for seed in tuning_seeds:
                q_table, _ = train(
                    config, learner, method, seed, settings.tuning_episodes
                )
                scores = []
                for student_index, student in enumerate(VALIDATION_TYPES):
                    reference = NumericalOracle(config, student, settings.oracle_grid)
                    result = evaluate(
                        config,
                        {"kind": "learner", "q": q_table},
                        student,
                        settings.eval_episodes,
                        480000 + seed * 100 + student_index,
                        reference,
                        learner.bins,
                    )
                    scores.append(result["objective_return"])
                tuning.append(
                    dict(
                        method=method,
                        profile=index,
                        seed=seed,
                        validation_objective=float(np.mean(scores)),
                        **asdict(learner),
                    )
                )
            print(
                f"Validated {method}: candidate {index + 1}/{len(PROFILES)}", flush=True
            )
    write_csv(directory / "tuning.csv", tuning)
    selected = {}
    for method in ("qlearning", "sarsa"):
        best_score = -float("inf")
        for index, learner in enumerate(PROFILES):
            scores = []
            for row in tuning:
                if row["method"] == method and row["profile"] == index:
                    scores.append(row["validation_objective"])
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                selected[method] = asdict(learner)
    write_json(directory / "selected.json", selected)
    print("Selected settings: " + json.dumps(selected), flush=True)
    return selected


# Run all seeds of the main benchmark or one sensitivity scenario.
# Choose budgets, run seeds in order, and save the completed result tables.
def run_benchmark(config, selected, settings, directory, populations, scenario=False):
    directory.mkdir(parents=True, exist_ok=True)
    if scenario:
        episodes = settings.scenario_episodes
        seeds = settings.scenario_seeds
        evaluation = min(settings.eval_episodes, 200)
        first_seed = 200
        matched = False
        baseline_episodes = min(100, evaluation)
    else:
        episodes = settings.episodes
        seeds = settings.seeds
        evaluation = settings.eval_episodes
        first_seed = 0
        matched = settings.matched
        baseline_episodes = min(150, evaluation)
    baseline = select_baselines(config, baseline_episodes, settings.oracle_grid)
    outputs = []
    for number in range(seeds):
        result = run_seed(
            config,
            selected,
            baseline,
            first_seed + number,
            episodes,
            evaluation,
            settings.oracle_grid,
            directory,
            populations,
            matched,
        )
        outputs.append(result)
        print(f"{directory.name}: seed {number + 1}/{seeds} complete", flush=True)
    save_results(
        directory,
        outputs,
        config,
        selected,
        baseline,
        episodes,
        evaluation,
        settings.oracle_grid,
        populations,
        matched,
    )


# Load settings and run tuning, the main benchmark, and optional pilots.
# Record success or failure and generate the report only after experiments finish.
def main(arguments):
    settings, config = read_settings(arguments)
    for name in (
        "episodes",
        "seeds",
        "eval_episodes",
        "tuning_episodes",
        "scenario_episodes",
        "scenario_seeds",
        "oracle_grid",
    ):
        if getattr(settings, name) < 1:
            raise ValueError(f"{name} must be positive")
    if settings.oracle_grid < 4:
        raise ValueError("oracle_grid must be at least 4")
    directory = Path(settings.output).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {directory}. Choose a new --output."
        )
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = dict(
        arguments=vars(settings),
        resolved={
            name: getattr(settings, name)
            for name in (
                "episodes",
                "seeds",
                "eval_episodes",
                "tuning_episodes",
                "scenario_episodes",
                "scenario_seeds",
                "oracle_grid",
            )
        },
        train_types=TRAIN_TYPES,
        validation_types=VALIDATION_TYPES,
        test_types=TEST_TYPES,
        profiles=[asdict(profile) for profile in PROFILES],
        python=platform.python_version(),
        numpy=np.__version__,
        status="running",
        reference="Approximate known-type continuous-model numerical reference; not a proven upper bound",
        selection="Mean validation objective only. Test population excluded from selection.",
    )
    write_json(directory / "protocol.json", protocol)
    try:
        selected = select_learners(config, settings, directory)
        populations = {"training": TRAIN_TYPES, "test": TEST_TYPES}
        run_benchmark(config, selected, settings, directory, populations)
        diagnostics = oracle_refinement(
            config, settings.oracle_grid, settings.eval_episodes
        )
        write_csv(directory / "oracle_refinement.csv", diagnostics)
        if settings.profile == "full" and not settings.skip_scenarios:
            populations = {"training": TRAIN_TYPES, "validation": VALIDATION_TYPES}
            for name, changes in SCENARIOS.items():
                scenario_config = replace(config, **changes)
                run_benchmark(
                    scenario_config,
                    selected,
                    settings,
                    directory / "scenarios" / name,
                    populations,
                    scenario=True,
                )
        protocol["status"] = "complete"
        protocol["wall_seconds"] = time.perf_counter() - started
        write_json(directory / "protocol.json", protocol)
        from report_results import generate

        generate(directory)
        print(f"Finished. Open {directory / 'REPORT.md'}", flush=True)
    except BaseException as error:
        protocol["status"] = "failed"
        protocol["error"] = str(error)
        write_json(directory / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).with_name("project.config")))
    p.add_argument("--profile", choices=("full", "smoke"))
    p.add_argument("--output")
    for name in (
        "episodes",
        "seeds",
        "eval-episodes",
        "tuning-episodes",
        "scenario-episodes",
        "scenario-seeds",
        "oracle-grid",
    ):
        p.add_argument("--" + name, type=int)
    p.add_argument(
        "--skip-scenarios", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument(
        "--matched",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add a controlled common-settings on/off-policy comparison",
    )
    main(p.parse_args())
