"""Tabular Q-learning and SARSA with matched configurable exploration."""

from dataclasses import dataclass
import math
import numpy as np
from study_env import StudyEnv


@dataclass(frozen=True)
class LearnerConfig:
    bins: int = 3
    alpha: float = 0.3
    alpha_floor: float = 0.01
    epsilon_final: float = 0.01
    decay_fraction: float = 0.85

    # Check settings before the project starts using them.
    # Reject invalid ranges and non-finite values with clear error messages.
    def __post_init__(self):
        if self.bins not in (3, 5):
            raise ValueError("Use 3 proposal bins or 5 refinement bins")
        values = [self.alpha, self.alpha_floor, self.epsilon_final, self.decay_fraction]
        for value in values:
            if not math.isfinite(value):
                raise ValueError("Hyperparameters must be finite")
        if (
            not 0 < self.alpha_floor <= self.alpha <= 1
            or not 0 <= self.epsilon_final <= 1
            or (not 0 < self.decay_fraction <= 1)
        ):
            raise ValueError("Invalid learner parameters")


# Learn a Q-table using Q-learning or SARSA over many sessions.
# Select actions, update the chosen Q value, and record training statistics.
def train(config, settings, method, seed, episodes):
    if method not in ("qlearning", "sarsa") or episodes < 1:
        raise ValueError("Invalid training request")
    environment = StudyEnv(config, settings.bins, 10000 + seed)
    random_generator = np.random.default_rng(seed)
    q_table = np.zeros((environment.encoder.size, 4))
    visit_counts = np.zeros_like(q_table, dtype=np.int32)
    learning_curves = []
    recent_episodes = []

    # Choose an action using the current exploration probability.
    # Explore randomly or choose randomly among the highest-valued actions.
    def choose(state, exploration):
        if random_generator.random() < exploration:
            return int(random_generator.integers(4))
        row = q_table[state]
        choices = np.flatnonzero(row == row.max())
        return int(choices[int(random_generator.integers(len(choices)))])

    for episode in range(episodes):
        epsilon = settings.epsilon_final + (1 - settings.epsilon_final) * max(
            0.0, 1 - episode / (episodes * settings.decay_fraction)
        )
        state = environment.reset()
        action = choose(state, epsilon)
        task_return = objective_return = shaped_return = 0.0
        for step in range(config.horizon):
            next_state, reward, done, info = environment.step(action)
            if not done:
                next_action = choose(next_state, epsilon)
            else:
                next_action = 0
            target = reward
            if not done:
                if method == "qlearning":
                    next_value = q_table[next_state].max()
                else:
                    next_value = q_table[next_state, next_action]
                target += config.gamma * next_value
            visit_counts[state, action] += 1
            alpha = max(
                settings.alpha_floor,
                settings.alpha / (1 + visit_counts[state, action] / 200) ** 0.6,
            )
            q_table[state, action] += alpha * (target - q_table[state, action])
            discount = config.gamma**step
            task_return += discount * info["task_reward"]
            objective_return += discount * info["objective_reward"]
            shaped_return += discount * reward
            state, action = (next_state, next_action)
            if done:
                break
        recent_episodes.append(
            (task_return, objective_return, shaped_return, float(info["burnout"]))
        )
        if (episode + 1) % 1000 == 0 or episode + 1 == episodes:
            values = np.mean(recent_episodes, axis=0)
            learning_curves.append(
                dict(
                    episode=episode + 1,
                    block_episodes=len(recent_episodes),
                    task_return=float(values[0]),
                    objective_return=float(values[1]),
                    shaped_return=float(values[2]),
                    burnout_rate=float(values[3]),
                    epsilon=epsilon,
                )
            )
            recent_episodes.clear()
    return (q_table, learning_curves)
