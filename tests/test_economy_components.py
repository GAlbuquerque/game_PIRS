"""Focused tests for the editable economy components."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from history import EconomicHistory
from economy import Economy
from event_engine import EventEngine
from events import GameEvent
from indicators import EconomicIndicators
from laws_of_motion import (
    MotionResult,
    ad_as_errors,
    find_curve_intersection,
    solve_ad_as,
)
from parameters import EconomyParameters
from utils import compute_real_interest_rate


class LawsOfMotionTests(unittest.TestCase):
    def test_real_interest_rate_uses_linear_approximation(self):
        self.assertEqual(compute_real_interest_rate(10.0, 6.0), 4.0)

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

    def test_numerical_solution_drives_both_equation_errors_to_zero(self):
        parameters = EconomyParameters()
        result = solve_ad_as(4.0, 1.0, 5.0, 0.1, -0.2, parameters)
        errors = ad_as_errors(
            result.inflation,
            result.output_growth,
            4.0,
            1.0,
            0.1,
            -0.2,
            parameters,
        )
        self.assertTrue(np.all(np.abs(errors) <= parameters.solver_tolerance))

    def test_numerical_solver_can_solve_a_nonlinear_equation(self):
        parameters = EconomyParameters()

        def nonlinear_errors(candidate):
            first_value, second_value = candidate
            return np.array([
                first_value**2 - 4.0,
                second_value**2 - 9.0,
            ])

        solution = find_curve_intersection(
            nonlinear_errors,
            initial_guess=[1.0, 1.0],
            parameters=parameters,
        )
        np.testing.assert_allclose(solution, [2.0, 3.0], atol=1e-7)


class HistoryTests(unittest.TestCase):
    def test_economy_uses_configured_minimum_inflation(self):
        parameters = EconomyParameters(minimum_inflation=-20.0)
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, 1),
            parameters=parameters,
        )
        motion = MotionResult(-30.0, 2.0, 5.0, 0.0, 2.0, -30.0)

        economy._commit_motion(motion, previous_inflation=2.0)

        self.assertEqual(economy.indicators.inflation_rate, -20.0)

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


class EventEngineTests(unittest.TestCase):
    def test_event_schedule_is_consumed_one_quarter_at_a_time(self):
        event = GameEvent(
            name="Test Event",
            description="A deterministic test event.",
            prob_terms=[],
            effects_schedule={"inflation": [1.0, 0.5]},
        )
        engine = EventEngine("central_banker", horizon=2, events=[])
        engine.enqueue(event)
        self.assertEqual(engine.consume_effects(), {"inflation": 1.0})
        self.assertEqual(engine.consume_effects(), {"inflation": 0.5})


if __name__ == "__main__":
    unittest.main()
