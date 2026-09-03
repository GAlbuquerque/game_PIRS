"""Tests for the model-settings simulation preview."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import _simulate_settings
from parameters import EconomyParameters


class SettingsSimulationTests(unittest.TestCase):
    def test_preview_includes_pre_player_and_player_quarters(self):
        result = _simulate_settings(
            EconomyParameters(),
            runs=2,
            turns=3,
            initialization_turns=2,
            scenario_name="Stable Economy",
        )

        frame = result["frame"]
        self.assertEqual(len(frame), 10)
        self.assertIn("Natural unemployment", frame.columns)
        self.assertIn("Reputation", frame.columns)
        self.assertTrue(frame["Reputation"].between(0, 1).all())
        for _, run in frame.groupby("Run"):
            self.assertEqual(run["Quarter"].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(
                run["Phase"].tolist(),
                ["Pre-player", "Pre-player"]
                + ["Player substitute"] * 3,
            )


if __name__ == "__main__":
    unittest.main()
