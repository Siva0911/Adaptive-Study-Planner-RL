"""Print a policy trajectory AFTER running the project; no model ships in the ZIP."""

import argparse
import json
from pathlib import Path
import numpy as np
from study_env import Config, StudyEnv, ACTIONS, TEST_TYPES


# Show a study session using a saved trained policy.
# Load its settings, choose a test student, and print every activity and state.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="results/qlearning_seed0.npz")
    parser.add_argument(
        "--student", type=int, choices=range(len(TEST_TYPES)), default=0
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    path = Path(args.policy)
    if not path.exists():
        parser.error(
            "Policy does not exist. Run run_project.py first, or select --policy."
        )
    saved = np.load(path, allow_pickle=False)
    config = Config(**json.loads(str(saved["config_json"])))
    learner = json.loads(str(saved["learner_json"]))
    environment = StudyEnv(config, learner["bins"], args.seed)
    observation = environment.reset(TEST_TYPES[args.student])
    print("Step  Action       New       Consolidated   Fatigue   Time left")
    for step in range(config.horizon):
        action = int(saved["q"][observation].argmax())
        observation, _, done, info = environment.step(action)
        print(
            f"{step + 1:4}  {ACTIONS[action]:9}  {environment.n:.5f}   {environment.c:.5f}        {environment.f:.5f}   {environment.remaining}"
        )
        if done:
            break
    print(f"Exam score={info['score']:.5f}; burnout={info['burnout']}")


if __name__ == "__main__":
    main()
