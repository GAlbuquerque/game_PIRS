"""Focused tests for the editable economy components."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from history import EconomicHistory
from economy import Economy
from indicators import EconomicIndicators
from laws_of_motion import solve_ad_as
from parameters import EconomyParameters


class LawsOfMotionTests(unittest.TestCase):
    def test_ad_and_as_intersect_and_okun_runs_afterward(self):
        parameters = EconomyParameters()
        result = solve_ad_as(4.0, 1.0, 5.0, 0.1, -0.2, parameters)

        self.assertAlmostEqual(result.output_growth, result.aggregate_demand)
        self.assertAlmostEqual(result.inflation, result.aggregate_supply)
        expected_unemployment = 5.0 - parameters.okun_coefficient * result.output_gap
        self.assertAlmostEqual(result.unemployment, expected_unemployment)

    def test_player_interest_rate_changes_demand(self):
        parameters = EconomyParameters()
        low = solve_ad_as(2, 1, 5, 0, 0, parameters)
        high = solve_ad_as(6, 1, 5, 0, 0, parameters)
        self.assertGreater(low.output_growth, high.output_growth)


class HistoryTests(unittest.TestCase):
    def test_random_history_retains_all_relevant_series(self):
        initial = EconomicIndicators(2, 5, 5, 2, 1)
        history = EconomicHistory.generate_random(
            6, initial, EconomyParameters(), np.random.default_rng(7)
        )
        self.assertEqual(len(history.entries), 6)
        self.assertEqual(len(history.series("gdp_growth")), 6)
        self.assertIn("events", history.to_frame().columns)

    def test_economy_keeps_prehistory_initial_state_and_new_quarters(self):
        initial = EconomicIndicators(2, 5, 5, 2, 1)
        economy = Economy(initial_state=initial, random_history_quarters=3)
        self.assertEqual([entry.quarter for entry in economy.history.entries], [-3, -2, -1, 0])
        economy.simulate_quarter()
        self.assertEqual(economy.history.entries[-1].quarter, 1)


if __name__ == "__main__":
    unittest.main()
