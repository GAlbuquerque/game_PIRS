"""Tests for portable, file-free in-progress game saves."""

import pathlib
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from game_code import decode_game_code, encode_game_code


class GameCodeTests(unittest.TestCase):
    @staticmethod
    def _started_app():
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(button for button in app.button if button.label == "Start Game").click().run()
        return app

    def test_game_state_survives_save_code_round_trip(self):
        app = self._started_app()
        next(widget for widget in app.text_input if widget.key == "rate_text").set_value(
            "3.75"
        ).run()
        next(button for button in app.button if button.label == "Next").click().run()
        original = app.session_state.economy

        restored, session = decode_game_code(
            encode_game_code(original, app.session_state)
        )

        self.assertEqual(restored.current_quarter, original.current_quarter)
        self.assertEqual(restored.interest_rate, 3.75)
        self.assertEqual(restored.reputation, original.reputation)
        self.assertEqual(restored.indicators, original.indicators)
        self.assertEqual(restored.history.entries, original.history.entries)
        self.assertEqual(restored.past_events, original.past_events)
        self.assertEqual(
            [dict(item) for item in restored.effect_queue],
            [dict(item) for item in original.effect_queue],
        )
        self.assertEqual(session["player_turn"], 2)

    def test_loaded_game_can_continue(self):
        app = self._started_app()
        saved_quarter = app.session_state.economy.current_quarter
        save_code = encode_game_code(app.session_state.economy, app.session_state)

        next(widget for widget in app.text_input if widget.key == "rate_text").set_value(
            "8.0"
        ).run()
        next(button for button in app.button if button.label == "Next").click().run()
        next(
            widget for widget in app.text_input if widget.key == "game_code_input"
        ).set_value(save_code).run()
        next(button for button in app.button if button.label == "Load game").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.economy.current_quarter, saved_quarter)
        self.assertEqual(len(app.success), 1)
        next(button for button in app.button if button.label == "Next").click().run()
        self.assertEqual(app.session_state.economy.current_quarter, saved_quarter + 1)

    def test_invalid_or_wrong_kind_of_code_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid PIRS-GAME1"):
            decode_game_code("PIRS2:{}")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            decode_game_code("PIRS-GAME1:not-json")


if __name__ == "__main__":
    unittest.main()
