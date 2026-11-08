import unittest
import numpy as np
from study_env import Config, TRAIN_TYPES, VALIDATION_TYPES, TEST_TYPES
from oracle import NumericalOracle
from evaluation import evaluate, aggregate


class AnalysisTests(unittest.TestCase):

    # Check simplex interpolation reproduces affine functions.
    # Interpolate a known linear function and compare it with its exact value.
    def test_simplex_interpolation_reproduces_affine_functions(self):
        oracle = NumericalOracle(Config(horizon=2), TRAIN_TYPES[0], grid=8)
        g = oracle.grid
        n, c, f = np.indices((g + 1, g + 1, g + 1)) / g
        value = 2 * n + 3 * c + 4 * f
        random_generator = np.random.default_rng(4)
        states = np.column_stack(
            (
                random_generator.dirichlet([1, 1, 1], 100)[:, :2],
                random_generator.random(100),
            )
        )
        interpolated = oracle.interpolate(value, states)
        np.testing.assert_allclose(
            interpolated, states @ np.array([2, 3, 4]), atol=1e-12
        )

    # Check one step oracle matches manual gaussian quadrature.
    # Calculate the one-step expectation independently and compare the planned value.
    def test_one_step_oracle_matches_manual_gaussian_quadrature(self):
        config = Config(horizon=1)
        student = TRAIN_TYPES[0]
        oracle = NumericalOracle(config, student, grid=8)
        nodes, weights = np.polynomial.hermite.hermgauss(7)
        gain = np.clip(student[0] + config.noise_std * np.sqrt(2) * nodes, 0, 1) * (
            1 - config.forgetting
        )
        expected = float(config.rho * gain @ (weights / np.sqrt(np.pi)))
        self.assertAlmostEqual(oracle.initial_value, expected, places=12)
        self.assertEqual(oracle.act((0, 0, 0), 1), 0)

    # Check no noise oracle policy return close to prediction.
    # Run the reference policy without noise and compare its return with its prediction.
    def test_no_noise_oracle_policy_return_close_to_prediction(self):
        config = Config(horizon=8, noise_std=0.0)
        oracle = NumericalOracle(config, TRAIN_TYPES[0], grid=16)
        result = evaluate(config, dict(kind="oracle"), TRAIN_TYPES[0], 30, 5, oracle)
        self.assertLess(abs(result["objective_return"] - oracle.initial_value), 0.04)

    # Check populations disjoint.
    # Compare student-type sets to ensure training, validation, and test groups differ.
    def test_populations_disjoint(self):
        self.assertFalse(set(TRAIN_TYPES) & set(VALIDATION_TYPES))
        self.assertFalse(set(TEST_TYPES) & (set(TRAIN_TYPES) | set(VALIDATION_TYPES)))

    # Check undefined reference ratios are preserved.
    # Use missing reference ratios and confirm that summaries do not invent a percentage.
    def test_undefined_reference_ratios_are_preserved(self):
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
        rows = []
        for seed in (0, 1):
            row = dict(
                seed=seed,
                population="test",
                method="qlearning",
                **dict.fromkeys(keys, 0.5)
            )
            row["reference_ratio"] = None
            rows.append(row)
        summary = aggregate(rows)[0]
        self.assertIsNone(summary["reference_ratio"])
        self.assertIsNone(summary["mean_meets_85_percent_reference"])


if __name__ == "__main__":
    unittest.main()
