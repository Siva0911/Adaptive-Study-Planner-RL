# Reinforcement Learning Study Planner — three-person project

Combine the CONTENTS of all three ZIPs into the SAME repository root. The files
are deliberately divided by ownership, so one part alone cannot run the project.
Do not create part1/, part2/, or part3/ subfolders inside the repository.

## Team ownership

| Owner | Responsibility | Python files |
|---|---|---|
| You / Part 1 | Agents, main workflow, configuration, integration | agents.py, run_project.py, test_main.py |
| Teammate 2 / Part 2 | Continuous simulator and numerical planner | study_env.py, oracle.py, test_environment.py |
| Teammate 3 / Part 3 | Evaluation, reporting, checking, demo | evaluation.py, report_results.py, check_results.py, demo.py, test_analysis.py |

The split is approximately one third of the Python lines per person, including
owned tests. File counts differ because some files are longer and more demanding.
See CONTRIBUTIONS.md for the detailed responsibilities and shared interfaces.

All functions and methods have 2–3 explanatory comment lines above them. The
code uses ordinary sequential loops. NumPy numerical planning is retained because
it is needed for interpolation and Gaussian integration; it is an approximate
known-type reference, not an exact continuous-state optimum.

## Set up AFTER all three parts are present

Use Python 3.10+ (3.12 tested). Open PowerShell in the combined repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
```

Run the installation check, then the complete project:

```powershell
.\.venv\Scripts\python.exe run_project.py --profile smoke --no-matched --output smoke_results
.\.venv\Scripts\python.exe check_results.py smoke_results
.\.venv\Scripts\python.exe run_project.py
.\.venv\Scripts\python.exe check_results.py new_results
.\.venv\Scripts\python.exe demo.py --policy new_results/sarsa_seed0.npz
```

The full run performs validation tuning, ten final seeds, matched Q-learning and
SARSA, numerical-reference diagnostics, and eight sensitivity pilots. Allow tens
of minutes on a laptop; sequential execution may take longer. project.config
controls budgets and continuous environment settings. Command-line options
such as --episodes or --output override the file. --workers is not supported.

Open new_results/REPORT.md, comparison.png, learning.png, and summary.csv after
completion. Existing nonempty output directories are rejected; use a new name
to repeat a run. Smoke outputs are not final performance evidence.

## GitHub and submission

Follow GIT_WORKFLOW.md for the exact sequence: create one empty remote, publish
Part 1, let each teammate contribute on their own branch, merge both pull
requests, then pull main and run the final combined code. The three ZIPs contain
no .git directory, environment, old results, models, charts, or caches.

Results are generated only after running. They are excluded from ordinary Git
commits by .gitignore. The guide explains how to produce a final code-plus-results
submission ZIP without including .git, .venv, or caches.

## Model and interpretation

Knowledge and fatigue evolve continuously. Study uses Gaussian noise, new
knowledge decays every step, and Review/Practice consolidate it. Only the
observations are binned. Burnout takes precedence at the time limit; terminal
shaping potential is zero for both endings. Recent practice uses three attempts.

Training, validation, and final test students are separate. Validation chooses
settings; final test students are excluded from that selection. The matched
learners share hyperparameters to support the on/off-policy comparison.

An 85% reference-return ratio is a research target, not classification accuracy
or a guaranteed exam mark. The numerical reference is approximate and is not a
proven upper bound. Report actual findings, including methods that lose to a
heuristic. DQN and real-student efficacy are outside the implemented scope.
