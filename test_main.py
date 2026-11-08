import unittest
import numpy as np
from study_env import Config
from agents import LearnerConfig, train


class TrainingTests(unittest.TestCase):

    # Check training and saved shapes.
    # Train both algorithms briefly and check table sizes, finite values, and curve logs.
    def test_training_and_saved_shapes(self):
        config = Config(horizon=8)
        for method in ("qlearning", "sarsa"):
            learner = LearnerConfig()
            q_table, curves = train(config, learner, method, 8, 50)
            self.assertEqual(q_table.shape, (3**3 * 9 * 3, 4))
            self.assertTrue(np.isfinite(q_table).all())
            self.assertTrue(np.any(q_table))
            self.assertEqual(curves[-1]["episode"], 50)
            self.assertAlmostEqual(
                curves[-1]["objective_return"], curves[-1]["shaped_return"]
            )


class ConfigurationTests(unittest.TestCase):

    # Check config file loads and command line overrides.
    # Read the smoke profile and confirm that explicit command-line settings take priority.
    def test_config_file_loads_and_command_line_overrides(self):
        from argparse import Namespace
        from pathlib import Path
        from run_project import read_settings

        arguments = Namespace(
            config=str(Path(__file__).with_name("project.config")),
            profile="smoke",
            output="example_output",
            matched=False,
            skip_scenarios=True,
            episodes=17,
            seeds=None,
            eval_episodes=None,
            tuning_episodes=None,
            scenario_episodes=None,
            scenario_seeds=None,
            oracle_grid=None,
        )
        settings, environment = read_settings(arguments)
        self.assertEqual(settings.episodes, 17)
        self.assertEqual(settings.seeds, 1)
        self.assertEqual(settings.output, "example_output")
        self.assertFalse(settings.matched)
        self.assertAlmostEqual(environment.forgetting, 0.06)

    # Check missing config has clear error.
    # Request a nonexistent configuration file and check the resulting error message.
    def test_missing_config_has_clear_error(self):
        from argparse import Namespace
        from run_project import read_settings

        with self.assertRaisesRegex(ValueError, "Configuration file does not exist"):
            read_settings(Namespace(config="missing-project.config"))


if __name__ == "__main__":
    unittest.main()
