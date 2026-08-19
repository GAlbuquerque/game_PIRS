"""Tests for portable, file-free model calibration passwords."""

import json
import pathlib
import sys
import unittest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import MODEL_PARAMETER_ORDER, _decode_settings_code, _encode_settings_code
from parameters import EconomyParameters


class SettingsCodeTests(unittest.TestCase):
    def test_password_schema_matches_economy_parameters(self):
        parameter_fields = set(EconomyParameters.__dataclass_fields__)
        self.assertEqual(set(MODEL_PARAMETER_ORDER) - parameter_fields, set())

    def test_settings_survive_password_round_trip(self):
        original = {
            "demand_interest_rate_pressure": 0.8125,
            "event_probability_scale": 1.75,
            "shock_std_devs": (0.1, 0.2, 0.3, 0.4),
        }
        restored = _decode_settings_code(_encode_settings_code(original))
        self.assertEqual(restored["demand_interest_rate_pressure"], 0.8125)
        self.assertEqual(restored["event_probability_scale"], 1.75)
        self.assertEqual(restored["shock_std_devs"], (0.1, 0.2, 0.3, 0.4))

    def test_password_is_deterministic_and_tied_to_its_settings(self):
        first = {"demand_interest_rate_pressure": 0.8125}
        second = {"demand_interest_rate_pressure": 0.25}

        self.assertEqual(_encode_settings_code(first), _encode_settings_code(first))
        self.assertNotEqual(_encode_settings_code(first), _encode_settings_code(second))
        self.assertEqual(
            _decode_settings_code(_encode_settings_code(first))[
                "demand_interest_rate_pressure"
            ],
            0.8125,
        )
        visible_settings = json.loads(_encode_settings_code(first).removeprefix("PIRS2:"))
        self.assertEqual(visible_settings["demand_interest_rate_pressure"], 0.8125)

    def test_password_prefix_is_case_insensitive(self):
        code = _encode_settings_code({})
        restored = _decode_settings_code("pirs2:" + code.removeprefix("PIRS2:"))
        self.assertEqual(restored["event_probability_scale"], 1.0)

    def test_password_rejects_a_missing_parameter(self):
        settings = json.loads(_encode_settings_code({}).removeprefix("PIRS2:"))
        del settings["inflation_target"]
        with self.assertRaisesRegex(ValueError, "expected parameters"):
            _decode_settings_code("PIRS2:" + json.dumps(settings))

    def test_editor_updates_and_restores_all_widget_values(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(button for button in app.button if button.label == "Settings").click().run()

        original_code = app.code[0].value
        widget_values = {
            "setting_demand_interest_rate_pressure": 0.8125,
            "setting_inflation_target": 3.25,
            "setting_shock_0": 0.11,
            "setting_shock_1": 0.22,
            "setting_shock_2": 0.33,
            "setting_shock_3": 0.44,
        }
        for key, value in widget_values.items():
            next(widget for widget in app.number_input if widget.key == key).set_value(
                value
            ).run()
        changed_code = app.code[0].value

        self.assertNotEqual(changed_code, original_code)
        self.assertEqual(
            _decode_settings_code(changed_code)["demand_interest_rate_pressure"],
            0.8125,
        )

        for key in widget_values:
            next(widget for widget in app.number_input if widget.key == key).set_value(
                0.01
            ).run()
        app.text_input[0].set_value(changed_code).run()
        next(button for button in app.button if button.label == "Apply code").click().run()

        for key, expected in widget_values.items():
            with self.subTest(key=key):
                widget = next(
                    widget for widget in app.number_input if widget.key == key
                )
                self.assertAlmostEqual(widget.value, expected)
                self.assertAlmostEqual(app.session_state[key], expected)
        self.assertEqual(len(app.success), 1)

    def test_legacy_password_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid PIRS2"):
            _decode_settings_code("PIRS1-OLD-CODE")


if __name__ == "__main__":
    unittest.main()
