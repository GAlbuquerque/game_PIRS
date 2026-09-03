"""Tests for portable, file-free model calibration passwords."""

import json
import pathlib
import sys
import unittest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from app import (
    MODEL_PARAMETER_ORDER,
    PARAMETER_EQUATIONS,
    _decode_settings_code,
    _encode_settings_code,
)
from parameters import EconomyParameters
from settings_code import (
    LEGACY_MODEL_PARAMETER_ORDER,
    PREVIOUS_CURRENT_MODEL_PARAMETER_ORDER,
    PREVIOUS_MODEL_PARAMETER_ORDER,
)


class SettingsCodeTests(unittest.TestCase):
    def test_start_game_completes_initialization_without_an_exception(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()

        next(button for button in app.button if button.label == "Start Game").click().run()

        self.assertTrue(app.session_state.game_started)
        self.assertEqual(len(app.exception), 0)

    def test_submitting_interest_rate_advances_without_widget_state_error(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(button for button in app.button if button.label == "Start Game").click().run()
        starting_quarter = app.session_state.economy.current_quarter

        next(widget for widget in app.text_input if widget.key == "rate_text").set_value(
            "4.25"
        ).run()
        next(button for button in app.button if button.label == "Next").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.session_state.economy.current_quarter, starting_quarter + 1
        )
        self.assertEqual(app.session_state.economy.interest_rate, 4.25)

    def test_background_settings_show_equilibrium_real_rate_equation(self):
        equation, explanation = PARAMETER_EQUATIONS["Background economy & shocks"]
        self.assertIn(r"r_t^n=r_{t-1}^n", equation)
        self.assertIn(r"\bar r^n", equation)
        self.assertIn("equilibrium real rate", explanation)

    def test_password_schema_matches_economy_parameters(self):
        parameter_fields = set(EconomyParameters.__dataclass_fields__)
        self.assertEqual(set(MODEL_PARAMETER_ORDER) - parameter_fields, set())

    def test_settings_survive_password_round_trip(self):
        original = {
            "output_gap_expectation_persistence": 0.8125,
            "event_probability_scale": 1.75,
            "shock_std_devs": (0.1, 0.2, 0.3, 0.4),
        }
        restored = _decode_settings_code(_encode_settings_code(original))
        self.assertEqual(restored["output_gap_expectation_persistence"], 0.8125)
        self.assertEqual(restored["event_probability_scale"], 1.75)
        self.assertEqual(restored["shock_std_devs"], (0.1, 0.2, 0.3, 0.4))

    def test_password_is_deterministic_and_tied_to_its_settings(self):
        first = {"output_gap_expectation_persistence": 0.8125}
        second = {"output_gap_expectation_persistence": 0.25}

        self.assertEqual(_encode_settings_code(first), _encode_settings_code(first))
        self.assertNotEqual(_encode_settings_code(first), _encode_settings_code(second))
        self.assertEqual(
            _decode_settings_code(_encode_settings_code(first))[
                "output_gap_expectation_persistence"
            ],
            0.8125,
        )
        visible_settings = json.loads(_encode_settings_code(first).removeprefix("PIRS2:"))
        self.assertEqual(visible_settings["output_gap_expectation_persistence"], 0.8125)

    def test_password_prefix_is_case_insensitive(self):
        code = _encode_settings_code({})
        restored = _decode_settings_code("pirs2:" + code.removeprefix("PIRS2:"))
        self.assertEqual(restored["event_probability_scale"], 1.0)

    def test_default_calibration_has_four_percent_unemployment_target(self):
        restored = _decode_settings_code(_encode_settings_code({}))
        self.assertEqual(restored["unemployment_target"], 4.0)
        self.assertEqual(restored["reputation_expectation_coefficient"], 0.2)
        self.assertEqual(restored["phillips_output_gap"], 0.1)
        self.assertEqual(restored["deflation_adjustment_ratio"], 0.8)
        self.assertEqual(restored["interest_rate_pressure_persistence"], 0.8)
        self.assertEqual(restored["intertemporal_elasticity_inverse"], 2.0)
        self.assertEqual(restored["equilibrium_real_rate_anchor"], 0.5)

    def test_password_rejects_a_missing_parameter(self):
        settings = json.loads(_encode_settings_code({}).removeprefix("PIRS2:"))
        del settings["inflation_target"]
        with self.assertRaisesRegex(ValueError, "expected parameters"):
            _decode_settings_code("PIRS2:" + json.dumps(settings))

    def test_password_rejects_negative_probability(self):
        settings = json.loads(_encode_settings_code({}).removeprefix("PIRS2:"))
        settings["event_probability_scale"] = -1
        with self.assertRaisesRegex(ValueError, "event probability scale cannot be negative"):
            _decode_settings_code("PIRS2:" + json.dumps(settings))

    def test_previous_pirs2_schema_receives_new_model_defaults(self):
        defaults = EconomyParameters()
        legacy_settings = {}
        for name in LEGACY_MODEL_PARAMETER_ORDER:
            legacy_settings[name] = (
                getattr(defaults, name)
                if hasattr(defaults, name)
                else 1.0
            )
        legacy_settings["demand_interest_rate_pressure"] = 0.75

        restored = _decode_settings_code(
            "PIRS2:" + json.dumps(legacy_settings)
        )

        self.assertEqual(restored["output_gap_expectation_persistence"], 0.8)
        self.assertNotIn("negative_gap_slope_ratio", restored)

    def test_code_without_unemployment_ceiling_receives_seventy_percent_default(self):
        defaults = EconomyParameters()
        previous_settings = {
            name: getattr(defaults, name)
            for name in PREVIOUS_CURRENT_MODEL_PARAMETER_ORDER
        }

        restored = _decode_settings_code("PIRS2:" + json.dumps(previous_settings))

        self.assertEqual(restored["maximum_unemployment"], 70.0)

    def test_immediately_previous_schema_migrates_deflation_ratio(self):
        defaults = EconomyParameters()
        previous_settings = {}
        for name in PREVIOUS_MODEL_PARAMETER_ORDER:
            previous_settings[name] = (
                getattr(defaults, name)
                if hasattr(defaults, name)
                else 1.0
            )
        previous_settings["deflation_supply_slope_ratio"] = 0.4

        restored = _decode_settings_code(
            "PIRS2:" + json.dumps(previous_settings)
        )

        self.assertEqual(restored["deflation_adjustment_ratio"], 0.8)
        self.assertNotIn("solver_tolerance", restored)

    def test_editor_updates_and_restores_all_widget_values(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(
            button for button in app.button if button.label == "Advanced Settings"
        ).click().run()

        unemployment_target = next(
            widget for widget in app.text_input
            if widget.key == "setting_unemployment_target"
        )
        self.assertEqual(unemployment_target.value, "4.0")
        self.assertEqual(app.session_state.settings_preview_runs, 100)
        self.assertEqual(app.session_state.settings_preview_turns, 100)
        equilibrium_rate = next(
            widget for widget in app.number_input
            if widget.key == "setting_equilibrium_real_rate_anchor"
        )
        self.assertEqual(equilibrium_rate.value, 0.5)
        for key in (
            "settings_preview_runs",
            "settings_preview_turns",
            "settings_preview_initialization_turns",
        ):
            widget = next(w for w in app.number_input if w.key == key)
            self.assertIsNone(widget.max)

        original_code = app.code[0].value
        widget_values = {
            "setting_output_gap_expectation_persistence": 0.8125,
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
            _decode_settings_code(changed_code)["output_gap_expectation_persistence"],
            0.8125,
        )

        for key in widget_values:
            next(widget for widget in app.number_input if widget.key == key).set_value(
                0.01
            ).run()
        next(
            widget for widget in app.text_input
            if widget.key == "settings_code_input"
        ).set_value(changed_code).run()
        next(button for button in app.button if button.label == "Apply code").click().run()

        for key, expected in widget_values.items():
            with self.subTest(key=key):
                widget = next(
                    widget for widget in app.number_input if widget.key == key
                )
                self.assertAlmostEqual(widget.value, expected)
                self.assertAlmostEqual(app.session_state[key], expected)
        self.assertEqual(len(app.success), 1)

    def test_advanced_settings_launches_selected_setup_and_rate_floor(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(
            button for button in app.button if button.label == "Advanced Settings"
        ).click().run()

        next(w for w in app.number_input if w.key == "setting_minimum_interest_rate").set_value(-0.5)
        next(w for w in app.selectbox if w.key == "advanced_difficulty").select("Principles")
        next(w for w in app.selectbox if w.key == "advanced_scenario").select("Stable Economy")
        next(w for w in app.selectbox if w.key == "advanced_mandate").select("Dual Mandate")
        app.run()
        next(button for button in app.button if button.label == "Play").click().run()

        self.assertTrue(app.session_state.game_started)
        self.assertEqual(app.session_state.difficulty, "principles")
        self.assertEqual(app.session_state.scenario_name, "Stable Economy")
        self.assertEqual(app.session_state.mandate, "dual_mandate")
        self.assertEqual(app.session_state.minimum_interest_rate, -0.5)

    def test_other_target_rejects_non_numeric_and_negative_values(self):
        app_path = pathlib.Path(__file__).parents[1] / "game_PIRS" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=10).run()
        next(
            button for button in app.button if button.label == "Advanced Settings"
        ).click().run()

        next(
            widget for widget in app.radio
            if widget.key == "setting_inflation_target_mode"
        ).set_value("Other").run()
        next(
            widget for widget in app.text_input
            if widget.key == "setting_inflation_target"
        ).set_value("not a number").run()
        next(button for button in app.button if button.label == "Play").click().run()
        self.assertIn("Inflation target must be a number.", [error.value for error in app.error])
        self.assertFalse(app.session_state.game_started)

        next(
            widget for widget in app.text_input
            if widget.key == "setting_inflation_target"
        ).set_value("-1").run()
        next(button for button in app.button if button.label == "Play").click().run()
        self.assertIn("Inflation target cannot be negative.", [error.value for error in app.error])
        self.assertFalse(app.session_state.game_started)

    def test_legacy_password_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid PIRS2"):
            _decode_settings_code("PIRS1-OLD-CODE")


if __name__ == "__main__":
    unittest.main()
