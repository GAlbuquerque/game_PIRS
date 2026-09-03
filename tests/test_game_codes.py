"""Tests for portable saved-game codes and their Streamlit controls."""

import pathlib
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import _decode_game_code, _encode_game_code
from economy import Economy


class GameCodeTests(unittest.TestCase):
    def test_game_survives_code_round_trip(self):
        economy = Economy(difficulty="central_banker", minimum_interest_rate=-0.5)
        economy.adjust_interest_rate(3.25)
        economy.trigger_player_event("quantitative_easing")
        economy.simulate_quarter()
        state = {
            "player_turn": 7,
            "in_term_quarter": 7,
            "mandate": "dual_mandate",
            "news_log": [{"quarter": 1, "name": "Test", "detail": "News"}],
        }

        code = _encode_game_code(economy, state)
        restored, restored_state = _decode_game_code(code)

        self.assertTrue(code.startswith("PIRSG1:"))
        self.assertEqual(restored.current_quarter, economy.current_quarter)
        self.assertEqual(restored.interest_rate, economy.interest_rate)
        self.assertEqual(restored.minimum_interest_rate, -0.5)
        self.assertEqual(restored.player_event_queue, economy.player_event_queue)
        self.assertEqual(restored.effect_queue, economy.effect_queue)
        self.assertEqual(restored.history.entries, economy.history.entries)
        self.assertEqual(restored.variables.history, economy.variables.history)
        self.assertEqual(restored_state, state)

    def test_code_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "not a valid"):
            _decode_game_code("not a save")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            _decode_game_code("PIRSG1:{")

    def test_start_menu_can_load_a_saved_game(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        source = Economy(difficulty="senior")
        source.interest_rate = 4.75
        code = _encode_game_code(source, {
            "news_log": [], "game_over": False, "player_turn": 3,
            "in_term_quarter": 3, "difficulty": "senior",
            "scenario_name": "Random", "mandate": "inflation_target",
            "dual_unemployment_target": 4.0, "inflation_target": 2.0,
            "initial_inflation": source.indicators.inflation_rate,
            "initial_unemployment": source.indicators.unemployment_rate,
            "graph_window_mode": "full", "graph_split_mode": False,
            "show_targets_on_graph": False, "show_end_dialog": False,
            "latest_fired": False,
        })
        app = AppTest.from_file(str(app_path), default_timeout=10).run()

        next(w for w in app.text_area if w.key == "game_code_input").set_value(code).run()
        next(b for b in app.button if b.label == "Load game").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state.game_started)
        self.assertEqual(app.session_state.economy.interest_rate, 4.75)
        self.assertEqual(app.session_state.player_turn, 3)


if __name__ == "__main__":
    unittest.main()
