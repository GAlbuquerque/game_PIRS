"""Tests for portable, file-free model calibration passwords."""

import pathlib
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
