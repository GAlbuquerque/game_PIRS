"""Regression tests for difficulty-aware charts and post-retirement actions."""

import json
import pathlib
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import _plot_histories
from economy import Economy


class RetirementUiTests(unittest.TestCase):
    app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"

    def test_natural_unemployment_only_appears_when_provided(self):
        charts = {}
        for difficulty in ("principles", "central_banker"):
            economy = Economy(difficulty=difficulty)
            charts[difficulty] = json.dumps(
                _plot_histories(
                    economy, "full", False, False, "inflation_target", 5, False
                ).to_dict()
            )

        self.assertIn("Natural unemployment", charts["principles"])
        self.assertNotIn("Natural unemployment", charts["central_banker"])

    def test_retirement_keeps_results_and_offers_navigation(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        next(button for button in app.button if button.label == "Start Game").click().run()

        app.session_state.retired = True
        app.session_state.game_over = True
        app.run()

        labels = {button.label for button in app.button}
        self.assertIn("Replay Same Scenario", labels)
        self.assertIn("Return to Start", labels)
        self.assertEqual([button.label for button in app.get("download_button")], ["Download Graph"])

        next(button for button in app.button if button.label == "Replay Same Scenario").click().run()
        self.assertFalse(app.session_state.retired)
        self.assertFalse(app.session_state.game_over)
        self.assertEqual(app.session_state.player_turn, 1)

    def test_return_to_start_leaves_retired_game(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        next(button for button in app.button if button.label == "Start Game").click().run()
        app.session_state.retired = True
        app.session_state.game_over = True
        app.run()

        next(button for button in app.button if button.label == "Return to Start").click().run()

        self.assertFalse(app.session_state.game_started)
        self.assertIn("Start Game", {button.label for button in app.button})


if __name__ == "__main__":
    unittest.main()
