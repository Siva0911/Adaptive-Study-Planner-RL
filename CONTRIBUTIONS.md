# Three-person responsibility plan

This assigns responsibility for understanding, reviewing, testing, and maintaining
existing project code. Each person should commit their actual work and explain
it in their own pull request. The allocation is not a claim of prior authorship.

| ZIP | Owner | Python lines | Approximate share |
|---|---|---:|---:|
| Part_1_Main_Integration.zip | You | 679 | 34.6% |
| Part_2_Environment_Planner.zip | Teammate 2 | 633 | 32.3% |
| Part_3_Evaluation_Reporting.zip | Teammate 3 | 650 | 33.1% |

Counts include code, comments, and tests, and measure source volume rather than
exact effort. The numerical planner carries mathematical work; integration
carries coordination work. The distribution is intentionally approximate.

## Part 1 — your main integration contribution

Files: agents.py, run_project.py, test_main.py, project.config, requirements.txt,
README.md, GIT_WORKFLOW.md, CONTRIBUTIONS.md, .gitignore.

Explain Q-learning/SARSA updates, exploration, terminal handling, validation-only
selection, run configuration, and the overall workflow. Maintain the three main
training/configuration tests. Create the repository, review both teammates'
pull requests, merge the parts, run the final experiments, check the evidence,
and prepare the submission. Own shared configuration and top-level documents.

## Part 2 — student environment and numerical planning

Files: study_env.py, oracle.py, test_environment.py.

Explain continuous learning, forgetting, review, practice history, fatigue,
burnout, observation encoding, shaping, and reproducible episode noise. Explain
the approximate known-type numerical planner and its limitations. Maintain the
ten environment tests. Coordinate any changes to observations or transitions
with both other owners; these are shared interfaces.

## Part 3 — analysis, validation, and presentation evidence

Files: evaluation.py, report_results.py, check_results.py, demo.py,
test_analysis.py.

Explain baselines, held-out evaluation, seed-level confidence intervals,
reference ratios, saved-data checks, numerical-reference tests, plots, report,
and the policy demo. Maintain five analysis tests. Interpret results honestly,
including failed hypotheses, and help prepare the final presentation evidence.

## Rules that keep the parts compatible

- All files go directly in the repository root. Imports expect that layout.
- Each ZIP owns different filenames; do not copy another member's whole folder
  over your clone. Shared documents/configuration belong to Part 1.
- Keep existing function/class names, observation encoding, configuration keys,
  saved Q-table metadata, CSV columns, and return fields unless all owners agree
  to update their callers and tests together.
- Do not change random seeds, student populations, model parameters, or reward
  definitions just to obtain a higher result. Configuration changes are visible
  in project.config and recorded in run metadata.
- Run all 18 tests and the smoke workflow after BOTH teammate branches are merged.
- Every new or changed function/method needs a clear 2–3-line summary above it.
- Parts depend on one another; an individual ZIP is not a standalone runnable app.
