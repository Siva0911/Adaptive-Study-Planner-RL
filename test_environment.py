import unittest
import numpy as np
from study_env import Config, StudyEnv, transition, TRAIN_TYPES
from oracle import NumericalOracle


class FixedRNG:

    # Prepare fixed random draws for controlled simulator tests.
    # Store one Gaussian value and one uniform value to return during a test.
    def __init__(self, z, u):
        self.z, self.u = (z, u)

    # Supply a fixed Gaussian draw for a deterministic unit test.
    # Return the stored test value instead of drawing a random number.
    def normal(self):
        return self.z

    # Supply a fixed uniform draw for a deterministic unit test.
    # Return the stored test value so practice outcomes are predictable.
    def random(self):
        return self.u


class EnvironmentTests(unittest.TestCase):

    # Check forgetting is continuous and accumulates.
    # Take repeated breaks and compare new knowledge with the exact decay formula.
    def test_forgetting_is_continuous_and_accumulates(self):
        env = StudyEnv(Config(), seed=4)
        env.reset(TRAIN_TYPES[0])
        env.n, env.c, env.f = (0.5, 0.2, 0.3)
        for t in range(5):
            env.step(3)
            self.assertAlmostEqual(env.n, 0.5 * 0.94 ** (t + 1), places=13)
            self.assertAlmostEqual(env.c, 0.2)
        self.assertLess(env.n, 0.4)

    # Check scalar simulator matches vector planning dynamics.
    # Compare all four simulated actions with the planner across random valid states.
    def test_scalar_simulator_matches_vector_planning_dynamics(self):
        random_generator = np.random.default_rng(3)
        config = Config()
        for _ in range(100):
            n, c = random_generator.dirichlet([1, 1, 1])[:2]
            f = float(random_generator.uniform(0, 0.99))
            student = TRAIN_TYPES[int(random_generator.integers(4))]
            for action in range(4):
                z = float(random_generator.normal())
                env = StudyEnv(config)
                env.reset(student)
                env.n, env.c, env.f = (n, c, f)
                env.rng = FixedRNG(z, 0.3)
                _, _, _, info = env.step(action)
                expected, burn = transition(
                    np.array([n, c, f]), action, student, config, z
                )
                np.testing.assert_allclose(env.state, expected, atol=1e-13)
                self.assertEqual(info["burnout"], bool(burn))
                self.assertGreaterEqual(env.n, 0)
                self.assertGreaterEqual(env.c, 0)
                self.assertLessEqual(env.n + env.c, 1 + 1e-12)

    # Check shaping telescopes completion and burnout.
    # Sum discounted rewards and check the initial-potential offset for both endings.
    def test_shaping_telescopes_completion_and_burnout(self):
        for action in (0, 3):
            config = Config(horizon=8, gamma=0.99)
            env = StudyEnv(config)
            env.reset(TRAIN_TYPES[-1])
            env.n, env.c = (0.2, 0.2)
            initial = env.potential()
            shaped = objective = 0.0
            for t in range(config.horizon):
                _, reward, done, info = env.step(action)
                shaped += config.gamma**t * reward
                objective += config.gamma**t * info["objective_reward"]
                if done:
                    break
            self.assertTrue(done)
            self.assertAlmostEqual(shaped, objective - initial)
            self.assertEqual(info["burnout"], action == 0)
            with self.assertRaises(RuntimeError):
                env.step(3)

    # Check seeded episode streams survive different termination times.
    # End paired sessions differently, then verify the next episode uses identical noise.
    def test_seeded_episode_streams_survive_different_termination_times(self):
        a, b = (StudyEnv(seed=8), StudyEnv(seed=8))
        for _ in range(3):
            self.assertEqual(a.reset(TRAIN_TYPES[-1]), b.reset(TRAIN_TYPES[-1]))
            while not a.done:
                a.step(0)
            while not b.done:
                b.step(3)
        a.reset(TRAIN_TYPES[0])
        b.reset(TRAIN_TYPES[0])
        self.assertEqual(a.step(0), b.step(0))

    # Check practice history is last three attempts.
    # Feed known successes and failures and confirm that older attempts are discarded.
    def test_practice_history_is_last_three_attempts(self):
        env = StudyEnv(Config(horizon=10))
        env.reset(TRAIN_TYPES[0])
        for u in (0.0, 0.0, 0.0, 1.0, 1.0, 1.0):
            env.f = 0.0
            env.rng = FixedRNG(0.0, u)
            env.step(2)
        self.assertEqual(env.practice_count, 3)
        self.assertEqual(env.practice_successes, 0)
        self.assertEqual(env.history, 1)

    # Check invalid parameters and actions.
    # Pass invalid settings and action numbers and require a clear exception.
    def test_invalid_parameters_and_actions(self):
        for kwargs in (
            dict(gamma=2),
            dict(noise_std=-1),
            dict(horizon=0),
            dict(forgetting=1),
        ):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        env = StudyEnv()
        env.reset()
        for action in (-1, 4, 1.5):
            with self.assertRaises(ValueError):
                env.step(action)

    # Check gaussian noise is not quantized.
    # Sample repeated study gains and check their spread, mean, and distinct values.
    def test_gaussian_noise_is_not_quantized(self):
        config = Config(horizon=1)
        env = StudyEnv(config, seed=43)
        observed = []
        for _ in range(3000):
            env.reset(TRAIN_TYPES[0])
            env.step(0)
            observed.append(env.n)
        self.assertEqual(len(set(observed)), len(observed))
        expected_mean = TRAIN_TYPES[0][0] * (1 - config.forgetting)
        expected_std = config.noise_std * (1 - config.forgetting)
        self.assertLess(abs(np.mean(observed) - expected_mean), 0.0025)
        self.assertLess(abs(np.std(observed) - expected_std), 0.002)

    # Check burnout precedes time completion.
    # Reach the fatigue limit on the last step and require burnout rather than a score.
    def test_burnout_precedes_time_completion(self):
        env = StudyEnv(Config(horizon=1))
        env.reset((0.3, 0.625))
        _, reward, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(info["burnout"])
        self.assertEqual(info["score"], 0.0)
        self.assertEqual(reward, -env.cfg.burnout_penalty)

    # Check review conserves knowledge before forgetting.
    # Disable forgetting and verify that review transfers knowledge without creating it.
    def test_review_conserves_knowledge_before_forgetting(self):
        config = Config(forgetting=0.0)
        env = StudyEnv(config)
        env.reset(TRAIN_TYPES[0])
        env.n, env.c, env.f = (0.5, 0.2, 0.3)
        env.step(1)
        self.assertAlmostEqual(env.n + env.c, 0.7)
        self.assertAlmostEqual(env.c, 0.2 + config.consolidation * 0.7 * 0.5)

    # Check objective reward includes fatigue penalty.
    # Check that the configured fatigue cost is subtracted from the task reward.
    def test_objective_reward_includes_fatigue_penalty(self):
        config = Config(horizon=1, fatigue_weight=0.1, shaped=False)
        env = StudyEnv(config)
        env.reset(TRAIN_TYPES[0])
        _, reward, _, info = env.step(0)
        self.assertAlmostEqual(reward, info["task_reward"] - 0.1 * env.f)
        oracle = NumericalOracle(config, TRAIN_TYPES[0], grid=8)
        self.assertGreater(oracle.initial_value, 0.0)


if __name__ == "__main__":
    unittest.main()
