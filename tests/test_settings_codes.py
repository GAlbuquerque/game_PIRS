"""Tests for portable, file-free model calibration passwords."""

import pathlib
import sys
import unittest

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import _decode_settings_code, _encode_settings_code


class SettingsCodeTests(unittest.TestCase):
    def test_settings_survive_password_round_trip(self):
        original = {
            "demand_real_rate": -0.8125,
            "event_probability_scale": 1.75,
            "shock_std_devs": (0.1, 0.2, 0.3, 0.4),
        }
        restored = _decode_settings_code(_encode_settings_code(original))
        self.assertEqual(restored["demand_real_rate"], -0.8125)
        self.assertEqual(restored["event_probability_scale"], 1.75)
        self.assertEqual(restored["shock_std_devs"], (0.1, 0.2, 0.3, 0.4))

    def test_password_is_case_and_separator_insensitive(self):
        code = _encode_settings_code({})
        restored = _decode_settings_code(code.lower().replace("-", " - "))
        self.assertEqual(restored["event_probability_scale"], 1.0)

    def test_password_detects_a_typo(self):
        code = _encode_settings_code({})
        corrupted = code[:-1] + ("A" if code[-1] != "A" else "B")
        with self.assertRaisesRegex(ValueError, "mistyped"):
            _decode_settings_code(corrupted)

    def test_editor_updates_and_restores_password_without_form_submission(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(button for button in app.button if button.label == "Settings").click().run()

        original_code = app.code[0].value
        real_rate = next(
            widget
            for widget in app.number_input
            if widget.key == "setting_demand_real_rate"
        )
        real_rate.set_value(-0.8125).run()
        changed_code = app.code[0].value

        self.assertNotEqual(changed_code, original_code)
        self.assertEqual(_decode_settings_code(changed_code)["demand_real_rate"], -0.8125)

        real_rate = next(
            widget
            for widget in app.number_input
            if widget.key == "setting_demand_real_rate"
        )
        real_rate.set_value(-0.25).run()
        app.text_input[0].set_value(changed_code).run()
        next(button for button in app.button if button.label == "Apply code").click().run()

        restored_real_rate = next(
            widget
            for widget in app.number_input
            if widget.key == "setting_demand_real_rate"
        )
        self.assertEqual(restored_real_rate.value, -0.8125)


if __name__ == "__main__":
    unittest.main()
