"""Tests for player-defined starting conditions."""

import pathlib
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import CUSTOM_SCENARIO


class CustomScenarioTests(unittest.TestCase):
    def test_start_menu_routes_custom_scenario_to_setup(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()

        next(widget for widget in app.radio if widget.label == "Scenario").set_value(
            CUSTOM_SCENARIO
        ).run()
        next(button for button in app.button if button.label == "Start Game").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.start_page, "custom")
        self.assertTrue(any(item.label == "t−1 inflation (%)" for item in app.number_input))
        event_menu = next(
            item for item in app.selectbox if item.label == "Event that fires in t"
        )
        self.assertNotIn("Demo Probability Event", event_menu.options)

    def test_custom_values_are_used_to_calculate_quarter_one(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(widget for widget in app.radio if widget.label == "Scenario").set_value(
            CUSTOM_SCENARIO
        ).run()
        next(button for button in app.button if button.label == "Start Game").click().run()
        inputs = {widget.key: widget for widget in app.number_input}
        inputs["custom_inflation"].set_value(5.0)
        inputs["custom_unemployment"].set_value(7.0)
        inputs["custom_interest_rate"].set_value(3.0)
        inputs["custom_interest_rate_pressure"].set_value(0.0)
        next(widget for widget in app.selectbox if widget.key == "custom_event").set_value(
            "Fiscal Surplus"
        ).run()
        next(
            button for button in app.button if button.label == "Start custom scenario"
        ).click().run()

        economy = app.session_state.economy
        self.assertEqual(economy.current_quarter, 2)
        self.assertEqual(economy.player_start_turn, 1)
        self.assertEqual(economy.history.entries[0].inflation_rate, 5.0)
        self.assertEqual(economy.history.entries[0].unemployment_rate, 7.0)
        self.assertEqual(economy.history.entries[0].interest_rate, 3.0)
        self.assertEqual(economy.history.entries[1].events, ("Fiscal Surplus",))
        self.assertEqual(app.session_state.news_log[0]["quarter"], 1)


if __name__ == "__main__":
    unittest.main()
