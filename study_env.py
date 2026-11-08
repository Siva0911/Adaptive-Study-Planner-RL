"""Continuous student dynamics; discretization is ONLY an agent observation choice."""

from dataclasses import dataclass, asdict
import math
import numpy as np

ACTIONS = ("Study", "Review", "Practice", "Break")
TRAIN_TYPES = ((0.22, 0.1), (0.22, 0.16), (0.34, 0.1), (0.34, 0.16))
VALIDATION_TYPES = ((0.26, 0.12), (0.3, 0.14))
TEST_TYPES = ((0.19, 0.11), (0.27, 0.17), (0.37, 0.135), (0.41, 0.195))


@dataclass(frozen=True)
class Config:
    horizon: int = 20
    forgetting: float = 0.06
    consolidation: float = 0.45
    recall: float = 0.18
    rho: float = 0.45
    burnout_penalty: float = 1.0
    noise_std: float = 0.025
    gamma: float = 1.0
    fatigue_weight: float = 0.0
    shaped: bool = True
    burnout_threshold: float = 1.0
    break_recovery: float = 0.3
    study_cost: float = 1.6
    practice_cost: float = 1.0
    review_cost: float = 0.65

    # Check settings before the project starts using them.
    # Reject invalid ranges and non-finite values with clear error messages.
    def __post_init__(self):
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        values = asdict(self)
        for value in values.values():
            if not math.isfinite(float(value)):
                raise ValueError("All parameters must be finite")
        if (
            not 0 <= self.forgetting < 1
            or not 0 <= self.recall <= 1
            or (not 0 <= self.consolidation <= 1)
        ):
            raise ValueError("Invalid forgetting/consolidation rates")
        if not 0 < self.rho < 1 or not 0 < self.gamma <= 1:
            raise ValueError("rho must be in (0,1); gamma in (0,1]")
        if not 0 < self.burnout_threshold <= 1 or self.burnout_penalty <= 0:
            raise ValueError("Invalid burnout parameters")
        if min(self.noise_std, self.fatigue_weight, self.break_recovery) < 0:
            raise ValueError("Noise, fatigue weight, recovery must be nonnegative")
        if not self.study_cost > self.practice_cost > self.review_cost > 0:
            raise ValueError("Fatigue costs must satisfy study > practice > review > 0")


# Check the hidden learning and fatigue rates for one student.
# Convert both rates to numbers and require each to be between zero and one.
def validate_student(student):
    learning_rate, fatigue_rate = map(float, student)
    if not (
        math.isfinite(learning_rate)
        and math.isfinite(fatigue_rate)
        and (0 < learning_rate <= 1)
        and (0 < fatigue_rate <= 1)
    ):
        raise ValueError("Student learning/fatigue rates must be in (0,1]")
    return (learning_rate, fatigue_rate)


# Calculate continuous next states for numerical planning.
# Apply one action, forgetting, physical limits, and burnout to NumPy arrays.
def transition(states, action, student, config, standard_noise=0.0):
    if action not in range(4):
        raise ValueError("Action must be 0..3")
    learning_rate, fatigue_rate = validate_student(student)
    states = np.asarray(states, dtype=float)
    new_knowledge, consolidated, fatigue = (
        states[..., 0],
        states[..., 1],
        states[..., 2],
    )
    if action == 0:
        next_new = (
            new_knowledge
            + learning_rate * (1 - fatigue) * (1 - new_knowledge - consolidated)
            + config.noise_std * np.asarray(standard_noise)
        )
        next_consolidated = consolidated
        next_fatigue = fatigue + fatigue_rate * config.study_cost
    elif action in (1, 2):
        gain = (
            (config.consolidation if action == 1 else config.recall)
            * (1 - fatigue)
            * new_knowledge
        )
        next_new = new_knowledge - gain
        next_consolidated = consolidated + gain
        next_fatigue = fatigue + fatigue_rate * (
            config.review_cost if action == 1 else config.practice_cost
        )
    else:
        next_new, next_consolidated = (new_knowledge, consolidated)
        next_fatigue = np.maximum(0.0, fatigue - config.break_recovery)
    next_consolidated = np.clip(next_consolidated, 0.0, 1.0)
    next_new = np.clip(next_new, 0.0, 1.0 - next_consolidated) * (1 - config.forgetting)
    next_new, next_consolidated, next_fatigue = np.broadcast_arrays(
        next_new, next_consolidated, next_fatigue
    )
    burnout = next_fatigue >= config.burnout_threshold - 1e-12
    return (
        np.stack((next_new, next_consolidated, np.minimum(next_fatigue, 1.0)), axis=-1),
        burnout,
    )


class ObservationEncoder:

    # Prepare the table-index layout for observable student states.
    # Store bin counts and horizon, then calculate the number of Q-table rows.
    def __init__(self, horizon, bins=3):
        if bins < 2 or horizon < 1:
            raise ValueError("Invalid encoder dimensions")
        self.horizon, self.bins = (horizon, bins)
        self.size = bins**3 * (horizon + 1) * 3

    # Convert an observed student state into one Q-table row number.
    # Combine knowledge bins, fatigue, remaining time, and practice history.
    def encode(self, state, remaining, history):
        if not 0 <= remaining <= self.horizon or history not in (0, 1, 2):
            raise ValueError("Invalid time/history")
        bins = self.bins
        new_knowledge, consolidated, fatigue = (
            min(bins - 1, max(0, int(float(x) * bins))) for x in state
        )
        return (
            ((new_knowledge * bins + consolidated) * bins + fatigue)
            * (self.horizon + 1)
            + remaining
        ) * 3 + history


class StudyEnv:

    # Prepare a continuous student simulator with reproducible randomness.
    # Store environment settings, observation bins, and the initial random seed.
    def __init__(self, config=Config(), bins=3, seed=0):
        self.cfg, self.encoder = (config, ObservationEncoder(config.horizon, bins))
        self.seed = int(seed)
        self.episode = 0
        self.done = True

    # Start a reproducible study session with a fixed hidden student type.
    # Reset knowledge and fatigue, then create an episode-specific random stream.
    def reset(self, student=None, seed=None):
        if seed is not None:
            self.seed, self.episode = (int(seed), 0)
        self.rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.episode])
        )
        self.episode += 1
        type_draw = int(self.rng.integers(len(TRAIN_TYPES)))
        self.student = validate_student(
            TRAIN_TYPES[type_draw] if student is None else student
        )
        self.n = self.c = self.f = 0.0
        self.remaining = self.cfg.horizon
        self.practice_results = []
        self.practice_count = 0
        self.practice_successes = 0
        self.history = 0
        self.done = False
        return self.observation()

    # Return the current continuous knowledge and fatigue values.
    # Expose new knowledge, consolidated knowledge, and fatigue as a tuple.
    @property
    def state(self):
        return (self.n, self.c, self.f)

    # Calculate the knowledge value used for exam scoring and shaping.
    # Count consolidated knowledge fully and new knowledge using the rho weight.
    def potential(self):
        return self.c + self.cfg.rho * self.n

    # Return the information that a tabular agent may observe.
    # Encode the current physical state, time budget, and practice summary.
    def observation(self):
        return self.encoder.encode(self.state, self.remaining, self.history)

    # Perform one study action and calculate its reward.
    # Update continuous knowledge, fatigue, history, and terminal conditions.
    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset after termination")
        if not isinstance(action, (int, np.integer)) or action not in range(4):
            raise ValueError("Action must be an integer 0..3")
        config = self.cfg
        previous_potential = self.potential()
        noise = float(self.rng.normal())
        practice_draw = float(self.rng.random())
        outcome = None
        if action == 2:
            probability = 1 / (1 + math.exp(-(5 * (self.n + self.c) - 3 * self.f - 2)))
            outcome = int(practice_draw < probability)
            self.practice_results.append(outcome)
            self.practice_results = self.practice_results[-3:]
            self.practice_count = len(self.practice_results)
            self.practice_successes = sum(self.practice_results)
            self.history = 1
            if self.practice_successes / self.practice_count >= 0.5:
                self.history = 2
        learning_rate, fatigue_rate = self.student
        new_knowledge, consolidated, fatigue = (self.n, self.c, self.f)
        if action == 0:
            new_knowledge += (
                learning_rate * (1 - fatigue) * (1 - new_knowledge - consolidated)
                + config.noise_std * noise
            )
            fatigue += fatigue_rate * config.study_cost
        elif action in (1, 2):
            gain = (
                (config.consolidation if action == 1 else config.recall)
                * (1 - fatigue)
                * new_knowledge
            )
            new_knowledge -= gain
            consolidated += gain
            fatigue += fatigue_rate * (
                config.review_cost if action == 1 else config.practice_cost
            )
        else:
            fatigue = max(0.0, fatigue - config.break_recovery)
        self.c = min(1.0, max(0.0, consolidated))
        self.n = min(1 - self.c, max(0.0, new_knowledge)) * (1 - config.forgetting)
        burnout = fatigue >= config.burnout_threshold - 1e-12
        self.f = min(1.0, fatigue)
        self.remaining -= 1
        self.done = burnout or self.remaining == 0
        if burnout:
            task_reward = -config.burnout_penalty
        else:
            task_reward = self.potential() if self.done else 0.0
        objective_reward = task_reward - config.fatigue_weight * self.f
        shaping = (
            config.gamma * (0.0 if self.done else self.potential()) - previous_potential
        )
        reward = objective_reward + (shaping if config.shaped else 0.0)
        info = dict(
            task_reward=task_reward,
            objective_reward=objective_reward,
            burnout=bool(burnout),
            fatigue=self.f,
            score=0.0 if burnout else self.potential(),
            practice_outcome=outcome,
            new_knowledge=self.n,
            consolidated_knowledge=self.c,
        )
        return (self.observation(), reward, self.done, info)
