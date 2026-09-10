"""Regression tests for difficulty-aware charts and post-retirement actions."""

import json
import pathlib
import sys
import unittest
from dataclasses import asdict

from streamlit.testing.v1 import AppTest, AppTestError

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

    def test_past_20_chart_excludes_player_marker_outside_window(self):
        economy = Economy(difficulty="central_banker")
        economy.player_start_turn = 1
        for _ in range(25):
            economy.simulate_quarter()

        expected_start = len(economy.variables.get_history("inflation_rate")) - 20
        for split_mode in (False, True):
            spec = _plot_histories(
                economy,
                "past20",
                split_mode,
                False,
                "inflation_target",
                5,
                False,
            ).to_dict()
            plotted_quarters = [
                row["Quarter"]
                for dataset in spec["datasets"].values()
                for row in dataset
                if "Quarter" in row
            ]

            self.assertEqual(min(plotted_quarters), expected_start)
            self.assertNotIn(economy.player_start_turn, plotted_quarters)

            panels = spec["hconcat"] if split_mode else [spec]
            for panel in panels:
                x_scale = panel["layer"][0]["encoding"]["x"]["scale"]
                self.assertEqual(
                    x_scale,
                    {"domain": [expected_start, expected_start + 19], "nice": False},
                )

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

        active_term_button = next(
            button for button in app.button if button.label == "Next"
        )
        quarter_before_queued_clicks = app.session_state.economy.current_quarter
        active_term_button.click().run()
        active_term_button.click().run()
        self.assertEqual(
            app.session_state.economy.current_quarter,
            quarter_before_queued_clicks + 2,
        )

        for _ in range(13):
            next(button for button in app.button if button.label == "Next").click().run()

        # Simulate multiple browser events queued from the same rendered button.
        # Once the first event opens the term dialog, its server-side guard must
        # reject later events even though they were submitted before the rerender.
        final_term_button = next(
            button for button in app.button if button.label == "Next"
        )
        quarter_before_final_click = app.session_state.economy.current_quarter
        final_term_button.click().run()
        final_term_button.click().run()

        current_quarter = app.session_state.economy.current_quarter
        current_news_count = len(app.session_state.news_log)
        self.assertEqual(current_quarter, quarter_before_final_click + 1)
        next_button = next(button for button in app.button if button.label == "Next")
        self.assertFalse(next_button.disabled)
        next_button.click().run()
        self.assertEqual(app.session_state.economy.current_quarter, current_quarter)
        self.assertIn("See numeric score", {item.label for item in app.expander})
        self.assertIn("term_loss", app.session_state.end_summary)
        formulas = " ".join(item.value for item in app.latex)
        self.assertIn(r"\mathrm{Inflation\_Loss}", formulas)
        self.assertIn(r"\mathrm{Unemployment\_Loss}", formulas)
        self.assertIn(
            r"\mathrm{Loss}=\mathrm{Inflation\_Loss}", formulas
        )
        self.assertTrue(
            any(
                "context only; not included in Loss" in item.value
                for item in app.markdown
            )
        )
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
