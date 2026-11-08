# Shared GitHub repository: start to final submission

Use ONE repository and three real contributors. The instructions below create
it in YOUR GitHub account; no remote repository has been created automatically.
Use a repository name such as rl-study-planner. Replace YOUR_USERNAME in every
URL with the GitHub account that owns that repository. Commands use PowerShell.

## 1. Everyone: initial setup

Install Git and Python 3.10+. Confirm both are available:

```powershell
git --version
python --version
```

In your own local project clone, use your own name and GitHub-verified email:

```powershell
git config user.name "Your Name"
git config user.email "your-verified-email@example.com"
```

Run these config commands AFTER git init or git clone below. They affect that
repository only. Sign into GitHub when Git's credential manager prompts you.
Each person uses their own account and credentials.

## 2. You: create the empty remote

On GitHub, create a repository named rl-study-planner (private is suitable for
course teamwork unless your instructor requires public access). Create it
EMPTY: do not initialize a README, .gitignore, license, template, or generated
code because Part 1 already supplies the initial files.

Open repository Settings, choose Collaborators, and invite your two teammates.
They must accept their invitations before pushing their branches.

## 3. You: push Part 1 first

Create a NEW local folder and open PowerShell in it:

```powershell
mkdir rl-study-planner
cd rl-study-planner
git init -b main
git config user.name "Your Name"
git config user.email "your-verified-email@example.com"
```

Extract Part_1_Main_Integration.zip into this folder. Its files must be directly
inside rl-study-planner, not inside a nested Part_1 folder. Check that agents.py,
run_project.py, and project.config are visible here. Then:

```powershell
git add agents.py run_project.py test_main.py project.config requirements.txt README.md GIT_WORKFLOW.md CONTRIBUTIONS.md .gitignore
git commit -m "Add main workflow, RL agents, and team integration setup"
git remote add origin https://github.com/YOUR_USERNAME/rl-study-planner.git
git push -u origin main
```

Part 1 is your initial contribution. The repository is intentionally incomplete
until Parts 2 and 3 are merged. Do not run the project or its full tests yet.
Keep this clone for the final pull and run.

## 4. Teammate 2: push the environment/planner branch

Use a NEW folder on teammate 2's own computer:

```powershell
git clone https://github.com/YOUR_USERNAME/rl-study-planner.git
cd rl-study-planner
git config user.name "Teammate Two Name"
git config user.email "teammate-two-verified-email@example.com"
git switch -c feature/environment-planner
```

Extract the CONTENTS of Part_2_Environment_Planner.zip into this clone's root.
Review the code and tests, then stage only this owned set:

```powershell
git add study_env.py oracle.py test_environment.py
git commit -m "Add continuous student simulator and numerical planner"
git push -u origin feature/environment-planner
```

On GitHub open a pull request: base main, compare feature/environment-planner.
Describe the files and actual work reviewed or changed. Ask you to review it.
A partial branch can still lack Part 3's modules, so the complete tests wait for
both merges.

## 5. Teammate 3: push the evaluation/reporting branch

Use a NEW folder on teammate 3's own computer:

```powershell
git clone https://github.com/YOUR_USERNAME/rl-study-planner.git
cd rl-study-planner
git config user.name "Teammate Three Name"
git config user.email "teammate-three-verified-email@example.com"
git switch -c feature/evaluation-reporting
```

Extract the CONTENTS of Part_3_Evaluation_Reporting.zip into this clone's root.
Review the code and tests, then:

```powershell
git add evaluation.py report_results.py check_results.py demo.py test_analysis.py
git commit -m "Add policy evaluation, result checks, and reporting"
git push -u origin feature/evaluation-reporting
```

Open a pull request: base main, compare feature/evaluation-reporting. Ask you to
review it. Both teammates may work from the initial Part 1 commit; their file
sets do not overlap, so the initial assembly should merge cleanly.

## 6. You: review and merge BOTH pull requests

Review the filenames and code on GitHub. Merge the environment/planner pull
request and then the evaluation/reporting pull request into main. Only files
owned by that part should be added. Keep the configuration and shared documents
from Part 1. Do not force-push or replace the repository with a ZIP upload.

After BOTH are merged, return to your original clone:

```powershell
git status
git switch main
git pull --ff-only origin main
```

Make sure git status has no uncommitted work before pulling. If you changed
files, commit the intended work on your own branch first. Verify all eleven
Python files are present. There should be no test_project.py from the older
unsplit layout; its tests are now divided between the three owners.

## 7. You: install, test, then run the final project

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe run_project.py --profile smoke --no-matched --output smoke_results
.\.venv\Scripts\python.exe check_results.py smoke_results
.\.venv\Scripts\python.exe run_project.py
.\.venv\Scripts\python.exe check_results.py new_results
.\.venv\Scripts\python.exe demo.py --policy new_results/sarsa_seed0.npz
```

The unit tests should discover 18 tests. The complete run generates reports,
charts, metrics, and models in new_results. Give this run enough time to finish;
read new_results/protocol.json for its recorded completion status. The generated
report is new_results/REPORT.md. Runtime depends on your computer.

If an output directory already exists and is nonempty, choose a new name, for
example --output second_run, and pass that same name to check_results.py. If you
change the model settings, report the recorded new configuration; do not relabel
old outputs as results of a new run.

## 8. You: prepare a final code-plus-results submission ZIP

After check_results.py succeeds, run this from the repository root:

```powershell
$submissionFiles = @(git ls-files)
$submissionFiles += "new_results"
Compress-Archive -LiteralPath $submissionFiles -DestinationPath ..\RL_Project_Final_Submission.zip -Force
```

This packages tracked source/documents plus the actual completed run. It excludes
.git, .venv, caches, and smoke_results because those are not tracked and were not
added to the list. If your final output has another name, replace new_results
above with that directory. Include your separately prepared course report and
slides if the instructor requires them. Open the ZIP and inspect its contents
before uploading it.

For a source-only ZIP from the final main commit, use instead:

```powershell
git archive --format=zip --output=..\RL_Project_Code_Only.zip main
```

This last command includes committed files only, not generated results.

## 9. Later fixes

Each owner starts a new branch from updated main, changes only their owned files,
commits, pushes, and opens another pull request. You review and merge it, pull
again, and rerun relevant tests. Reuse neither an old result directory nor an
old result claim after changing code. A branch that needs another owner's API
change should coordinate that change explicitly.

## Official GitHub references

- [Create a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [Add local code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [Invite collaborators](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)
- [Create a pull request](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request)
