"""Regression tests for difficulty-aware charts and post-retirement actions."""

import json
import pathlib
import sys
import unittest
from dataclasses import asdict

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import CUSTOM_SCENARIO, _plot_histories
from economy import Economy


class RetirementUiTests(unittest.TestCase):
    app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"

    @staticmethod
    def _parameter_values(parameters):
        """Compare parameter values independent of JSON tuple/array normalization."""
        return json.loads(
            json.dumps(asdict(parameters), default=lambda value: value.tolist())
        )

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
        app.session_state.model_settings = {
            "inflation_target": 3.25,
            "event_probability_scale": 0.4,
        }
        app.session_state.minimum_interest_rate = -0.75
        next(button for button in app.button if button.label == "Start Game").click().run()

        app.session_state.retired = True
        app.session_state.game_over = True
        app.run()

        initial_economy = app.session_state.economy
        initial_parameters = self._parameter_values(initial_economy.parameters)
        initial_history = list(initial_economy.history.entries)
        initial_difficulty = app.session_state.difficulty
        initial_scenario = app.session_state.scenario_name
        initial_model_settings = dict(app.session_state.model_settings)
        initial_minimum_rate = initial_economy.minimum_interest_rate

        labels = {button.label for button in app.button}
        self.assertIn("Play Again", labels)
        self.assertIn("Return to Start", labels)
        self.assertEqual(list(app.get("download_button")), [])
        self.assertTrue(any("keeps all settings" in caption.value for caption in app.caption))

        initial_economy.adjust_interest_rate(11.0)
        initial_economy.simulate_quarter()
        next(button for button in app.button if button.label == "Play Again").click().run()
        self.assertFalse(app.session_state.retired)
        self.assertFalse(app.session_state.game_over)
        self.assertEqual(app.session_state.player_turn, 1)
        self.assertEqual(app.session_state.difficulty, initial_difficulty)
        self.assertEqual(app.session_state.scenario_name, initial_scenario)
        self.assertEqual(dict(app.session_state.model_settings), initial_model_settings)
        self.assertEqual(
            app.session_state.economy.minimum_interest_rate,
            initial_minimum_rate,
        )
        self.assertEqual(
            self._parameter_values(app.session_state.economy.parameters),
            initial_parameters,
        )
        self.assertEqual(app.session_state.economy.history.entries, initial_history)

    def test_return_to_start_leaves_retired_game(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        next(button for button in app.button if button.label == "Start Game").click().run()
        app.session_state.retired = True
        app.session_state.game_over = True
        app.run()

        next(button for button in app.button if button.label == "Return to Start").click().run()

        self.assertFalse(app.session_state.game_started)
        self.assertIn("Start Game", {button.label for button in app.button})

    def test_continuing_starts_a_fresh_term_boundary(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        next(button for button in app.button if button.label == "Start Game").click().run()
        for _ in range(16):
            next(button for button in app.button if button.label == "Next").click().run()

        current_quarter = app.session_state.economy.current_quarter
        current_news_count = len(app.session_state.news_log)
        next(
            button for button in app.button if button.label == "Continue Playing"
        ).click().run()

        self.assertEqual(app.session_state.in_term_quarter, 1)
        self.assertEqual(app.session_state.term_start_idx, current_quarter)
        self.assertEqual(app.session_state.term_start_news_idx, current_news_count)
        self.assertEqual(
            app.session_state.initial_inflation,
            app.session_state.economy.indicators.inflation_rate,
        )
        self.assertEqual(
            app.session_state.initial_unemployment,
            app.session_state.economy.indicators.unemployment_rate,
        )

    def test_play_again_preserves_custom_setup(self):
        app = AppTest.from_file(str(self.app_path), default_timeout=20).run()
        next(widget for widget in app.radio if widget.label == "Difficulty").set_value(
            "Senior"
        )
        next(widget for widget in app.radio if widget.label == "Scenario").set_value(
            CUSTOM_SCENARIO
        ).run()
        next(button for button in app.button if button.label == "Start Game").click().run()

        inputs = {widget.key: widget for widget in app.number_input}
        inputs["custom_inflation"].set_value(6.25)
        inputs["custom_unemployment"].set_value(8.5)
        inputs["custom_interest_rate"].set_value(4.75)
        next(button for button in app.button if button.label == "Start custom scenario").click().run()
        expected_history = list(app.session_state.economy.history.entries)
        expected_parameters = self._parameter_values(app.session_state.economy.parameters)

        app.session_state.retired = True
        app.session_state.game_over = True
        app.run()
        next(button for button in app.button if button.label == "Play Again").click().run()

        self.assertEqual(app.session_state.scenario_name, CUSTOM_SCENARIO)
        self.assertEqual(app.session_state.difficulty, "senior")
        self.assertEqual(
            self._parameter_values(app.session_state.economy.parameters),
            expected_parameters,
        )
        self.assertEqual(app.session_state.economy.history.entries, expected_history)


if __name__ == "__main__":
    unittest.main()
