"""Known-type approximate finite-horizon planning on the CONTINUOUS simulator.

Only the planner's value function is meshed. Simulator trajectories never round.
Simplex interpolation and Gauss-Hermite quadrature introduce approximation error;
this is a numerical reference, NOT a proven continuous-state upper bound.
"""

import math
import numpy as np
from study_env import transition, validate_student


class NumericalOracle:

    # Build a known-type numerical planning reference.
    # Use Gaussian quadrature and backward planning on an interpolated value mesh.
    def __init__(self, config, student, grid=16, quadrature=7):
        if grid < 4 or quadrature < 3:
            raise ValueError("Use grid >=4 and quadrature >=3")
        self.cfg, self.student = (config, validate_student(student))
        self.grid = grid
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature)
        self.noise = np.sqrt(2) * nodes
        self.weights = weights / np.sqrt(np.pi)
        grid_coordinates = np.array(
            [
                (n, c, fatigue_position)
                for n in range(grid + 1)
                for c in range(grid + 1 - n)
                for fatigue_position in range(grid + 1)
            ],
            dtype=int,
        )
        self.coords = grid_coordinates
        states = grid_coordinates.astype(float) / grid
        states[:, 2] *= config.burnout_threshold
        states[:, 2] = np.minimum(states[:, 2], config.burnout_threshold - 1e-10)
        self.states = states
        self.values = np.zeros((config.horizon + 1, grid + 1, grid + 1, grid + 1))
        self.q_values = np.zeros((config.horizon + 1, grid + 1, grid + 1, grid + 1, 4))
        self.transitions = []
        for action in range(4):
            if action == 0:
                noise = self.noise
            else:
                noise = np.array([0.0])
            if action == 0:
                weight = self.weights
            else:
                weight = np.ones(1)
            next_states, burnout = transition(
                states[:, None, :], action, student, config, noise[None, :]
            )
            self.transitions.append(
                (next_states, burnout, weight, self.interpolation_indices(next_states))
            )
        for remaining in range(1, config.horizon + 1):
            action_values = []
            for next_states, burnout, weight, indices in self.transitions:
                score = next_states[..., 1] + config.rho * next_states[..., 0]
                raw = np.where(
                    burnout, -config.burnout_penalty, score if remaining == 1 else 0.0
                )
                action_scores = raw - config.fatigue_weight * next_states[..., 2]
                if remaining > 1:
                    future = self.interpolate(
                        self.values[remaining - 1], indices=indices
                    )
                    action_scores += config.gamma * np.where(burnout, 0.0, future)
                action_values.append(action_scores @ weight)
            self.values[remaining][tuple(grid_coordinates.T)] = np.max(
                action_values, axis=0
            )
            self.q_values[remaining][tuple(grid_coordinates.T)] = np.asarray(
                action_values
            ).T
        self.initial_value = float(self.values[config.horizon, 0, 0, 0])

    # Locate nearby planning-grid points for continuous states.
    # Find triangle weights for knowledge and neighboring fatigue-grid levels.
    def interpolation_indices(self, states):
        states_array = np.asarray(states, dtype=float)
        new_position = np.clip(states_array[..., 0], 0.0, 1.0) * self.grid
        consolidated_position = np.clip(states_array[..., 1], 0.0, 1.0) * self.grid
        fatigue_position = (
            np.clip(states_array[..., 2] / self.cfg.burnout_threshold, 0.0, 1.0)
            * self.grid
        )
        new_index = np.floor(new_position + 1e-12).astype(int)
        consolidated_index = np.floor(consolidated_position + 1e-12).astype(int)
        new_index, consolidated_index = (
            np.minimum(new_index, self.grid),
            np.minimum(consolidated_index, self.grid),
        )
        new_fraction, consolidated_fraction = (
            np.maximum(0.0, new_position - new_index),
            np.maximum(0.0, consolidated_position - consolidated_index),
        )
        upper = new_fraction + consolidated_fraction > 1.0
        n0 = np.where(upper, new_index + 1, new_index)
        c0 = np.where(upper, consolidated_index + 1, consolidated_index)
        n1 = new_index + 1
        c1 = consolidated_index
        n2 = new_index
        c2 = consolidated_index + 1
        w0 = np.where(
            upper,
            new_fraction + consolidated_fraction - 1.0,
            1 - new_fraction - consolidated_fraction,
        )
        w1 = np.where(upper, 1 - consolidated_fraction, new_fraction)
        w2 = np.where(upper, 1 - new_fraction, consolidated_fraction)
        lower_fatigue = np.minimum(
            np.floor(fatigue_position).astype(int), self.grid - 1
        )
        fraction = fatigue_position - lower_fatigue
        vertices = [
            (np.clip(n, 0, self.grid), np.clip(c, 0, self.grid), weight)
            for n, c, weight in ((n0, c0, w0), (n1, c1, w1), (n2, c2, w2))
        ]
        return (vertices, lower_fatigue, fraction)

    # Estimate a value between the numerical planning-grid points.
    # Combine nearby values using knowledge-triangle and fatigue weights.
    def interpolate(self, value, states=None, indices=None):
        if indices is not None:
            vertices, lower_fatigue, fraction = indices
        else:
            vertices, lower_fatigue, fraction = self.interpolation_indices(states)
        answer = np.zeros_like(fraction)
        for n, c, weight in vertices:
            answer += weight * (
                (1 - fraction) * value[n, c, lower_fatigue]
                + fraction * value[n, c, lower_fatigue + 1]
            )
        return answer

    # Choose the numerical-reference action for a continuous state.
    # Interpolate planned action values and return the highest-valued action.
    def act(self, state, remaining):
        if not 1 <= remaining <= self.cfg.horizon:
            raise ValueError("Oracle requires nonterminal remaining time")
        grid_size = self.grid
        new_position, consolidated_position, z = (
            state[0] * grid_size,
            state[1] * grid_size,
            state[2] * grid_size / self.cfg.burnout_threshold,
        )
        new_index, consolidated_index = (
            min(grid_size, math.floor(new_position)),
            min(grid_size, math.floor(consolidated_position)),
        )
        new_fraction, consolidated_fraction = (
            new_position - new_index,
            consolidated_position - consolidated_index,
        )
        lower_fatigue = min(grid_size - 1, max(0, math.floor(z)))
        fatigue_fraction = min(1.0, max(0.0, z - lower_fatigue))
        if new_fraction + consolidated_fraction <= 1:
            vertices = (
                (
                    new_index,
                    consolidated_index,
                    1 - new_fraction - consolidated_fraction,
                ),
                (min(grid_size, new_index + 1), consolidated_index, new_fraction),
                (
                    new_index,
                    min(grid_size, consolidated_index + 1),
                    consolidated_fraction,
                ),
            )
        else:
            vertices = (
                (
                    min(grid_size, new_index + 1),
                    min(grid_size, consolidated_index + 1),
                    new_fraction + consolidated_fraction - 1,
                ),
                (
                    min(grid_size, new_index + 1),
                    consolidated_index,
                    1 - consolidated_fraction,
                ),
                (new_index, min(grid_size, consolidated_index + 1), 1 - new_fraction),
            )
        action_scores = np.zeros(4)
        values = self.q_values[remaining]
        for n, c, weight in vertices:
            action_scores += weight * (
                (1 - fatigue_fraction) * values[n, c, lower_fatigue]
                + fatigue_fraction * values[n, c, lower_fatigue + 1]
            )
        return int(action_scores.argmax())
