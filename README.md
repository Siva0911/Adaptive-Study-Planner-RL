# Adaptive Study Planner Using Reinforcement Learning

This project uses reinforcement learning to plan a simulated student’s study session. At each step, an agent chooses whether to **Study, Review, Practice, or take a Break**, balancing learning progress against forgetting, fatigue, and limited time.

The project compares **Q-learning** and **SARSA** with random actions, fixed schedules, threshold-based rules, and a known-type numerical planning reference.

## How the Environment Works

Each study session tracks:

- **New knowledge:** recently learned material that gradually decays.
- **Consolidated knowledge:** retained material strengthened through review and practice.
- **Fatigue:** increases during learning activities and decreases during breaks.
- **Remaining time:** every activity consumes one step.
- **Recent practice performance:** summarizes the latest three practice attempts.

Students have different learning and fatigue rates, which remain hidden from the agent. Physical knowledge and fatigue values evolve continuously; only the agent’s observations are divided into bins.

A session ends when time runs out or fatigue reaches the burnout threshold. The final exam score is:

```text
Exam score = Consolidated knowledge + ρ × New knowledge
```

Burnout receives a negative reward and contributes zero to the reported exam-score average.

## Code Flow

```text
Load project.config
        ↓
Create the continuous student environment
        ↓
Train candidate Q-learning and SARSA agents
        ↓
Select settings using validation students
        ↓
Train the selected agents across multiple seeds
        ↓
Evaluate agents and baselines on training and held-out students
        ↓
Run sensitivity experiments and numerical-reference checks
        ↓
Generate metrics, charts, trained policies, and a report
```

Final test students are excluded from parameter selection.

## Main Files

| File | Purpose |
|---|---|
| `study_env.py` | Student dynamics, observations, rewards, forgetting, and burnout |
| `agents.py` | Q-learning and SARSA training |
| `oracle.py` | Approximate planning reference with known student characteristics |
| `evaluation.py` | Baselines, policy evaluation, and statistical summaries |
| `run_project.py` | Configuration, tuning, training, and experiment workflow |
| `report_results.py` | Generated findings and charts |
| `check_results.py` | Saved-result completeness and consistency checks |
| `demo.py` | Displays a session using a trained policy |
| `test_main.py` | Training and configuration tests |
| `test_environment.py` | Student-environment tests |
| `test_analysis.py` | Numerical-planning and evaluation tests |
| `project.config` | Experiment budgets and environment settings |

## Experiments

The complete workflow includes:

- Validation-based parameter selection.
- Ten main training seeds with 40,000 episodes per learner per seed.
- Evaluation using 500 episodes per student type per seed.
- A matched-settings Q-learning/SARSA comparison.
- Sensitivity experiments for session length, reward shaping, fatigue penalties, burnout penalties, noise, and discounting.
- Mesh and Gaussian-quadrature refinement checks for the numerical reference.

## Result Summary

The completed continuous-model run produced the following held-out test results:

| Method | Mean exam score | Burnout rate | Return relative to numerical reference |
|---|---:|---:|---:|
| Q-learning | 0.6323 | 4.31% | 80.95% |
| SARSA | 0.6711 | 0.765% | 90.31% |
| Threshold heuristic | 0.6922 | 0% observed | 93.98% |
| Fixed schedule | 0.4839 | 25.00% | 36.88% |

SARSA achieved higher test return and lower burnout than the separately tuned Q-learning agent. The threshold heuristic outperformed both tuned learners on this test population.

The reference percentage measures **objective return relative to an approximate numerical planning value**. It is not classification accuracy or a percentage exam mark. The numerical reference is not a proven upper bound.

## Running the Project

Use Python 3.10 or newer. From the project folder in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
```

Run a short installation check:

```powershell
.\.venv\Scripts\python.exe run_project.py --profile smoke --no-matched --output smoke_results
.\.venv\Scripts\python.exe check_results.py smoke_results
```

Run the complete experiment:

```powershell
.\.venv\Scripts\python.exe run_project.py
.\.venv\Scripts\python.exe check_results.py new_results
```

Display a trained policy’s study session:

```powershell
.\.venv\Scripts\python.exe demo.py --policy new_results/sarsa_seed0.npz
```

Use a new output directory when repeating an experiment. Existing nonempty directories are rejected to prevent mixing results.

## Generated Outputs

The completed run creates:

- `summary.csv`: population-level results and confidence intervals.
- `metrics.csv`: per-student, per-seed evaluation metrics.
- `learning_curves.csv`: training returns and burnout frequencies.
- `oracle_refinement.csv`: numerical-reference sensitivity checks.
- `comparison.png` and `learning.png`: evaluation and training charts.
- `REPORT.md`: generated findings and interpretation.
- `.npz` files: trained Q-tables and their configurations.
- `scenarios/`: sensitivity-experiment results.

## Scope

This project studies reinforcement learning in a controlled, hand-designed simulator. Hidden student characteristics and compressed observations make the learning problem partially observable. The findings describe the simulated populations and do not establish effectiveness for real students.
