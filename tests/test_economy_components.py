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
    calculate_demand_intercept,
    calculate_vertical_supply_output_growth,
    find_curve_intersection,
    solve_ad_as,
)
from parameters import EconomyParameters
from utils import compute_real_interest_rate


class LawsOfMotionTests(unittest.TestCase):
    def test_demand_intercept_uses_rate_gaps_and_potential_growth(self):
        parameters = EconomyParameters(
            potential_growth=2.0,
            demand_intercept_weight_10=-0.1,
            demand_intercept_weight_20=-0.2,
        )
        rates = list(range(1, 21))
        equilibrium_rates = [1.0] * 20

        result = calculate_demand_intercept(rates, equilibrium_rates, parameters)

        gaps = np.asarray(rates) - equilibrium_rates
        expected = 2.0 - 0.1 * np.mean(gaps[-10:]) - 0.2 * np.mean(gaps[-20:])
        self.assertAlmostEqual(result, expected)

    def test_demand_intercept_rejects_mismatched_rate_histories(self):
        with self.assertRaisesRegex(ValueError, "must have equal length"):
            calculate_demand_intercept([1.0, 2.0], [1.0], EconomyParameters())

    def test_vertical_supply_output_is_derived_from_okuns_law(self):
        parameters = EconomyParameters(
            potential_growth=2.0,
            okun_coefficient=0.4,
            vertical_supply_unemployment=2.0,
        )

        output_growth = calculate_vertical_supply_output_growth(5.0, parameters)

        self.assertAlmostEqual(output_growth, 9.5)
        self.assertAlmostEqual(
            5.0
            - parameters.okun_coefficient
            * (output_growth - parameters.potential_growth),
            parameters.vertical_supply_unemployment,
        )

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

    def test_supply_intercept_blends_target_and_lagged_inflation(self):
        parameters = EconomyParameters(phillips_output_gap=0.0)
        result = solve_ad_as(
            4, 1, 8, 0, 0, parameters,
            previous_inflation=6,
            target_inflation=2,
            reputation=0.8,
            natural_unemployment=5,
        )

        # alpha = R / 4 = .2, so beta_0,pi = .2(2) + .8(6) = 5.2.
        self.assertAlmostEqual(result.inflation, 5.2)

    def test_unemployment_intercept_is_the_natural_rate(self):
        parameters = EconomyParameters()
        result = solve_ad_as(
            4, 1, 20, 0, 0, parameters, natural_unemployment=4.5
        )

        self.assertAlmostEqual(
            result.unemployment,
            4.5 - parameters.okun_coefficient * result.output_gap,
        )

    def test_vertical_supply_caps_output_at_two_percent_unemployment(self):
        parameters = EconomyParameters(demand_intercept=20)
        natural_rate = 5.0
        capacity = parameters.potential_growth + (
            natural_rate - parameters.vertical_supply_unemployment
        ) / parameters.okun_coefficient
        result = solve_ad_as(
            0, 0, natural_rate, 0, 0, parameters,
            natural_unemployment=natural_rate,
            vertical_supply_output_growth=capacity,
        )

        self.assertAlmostEqual(result.output_growth, capacity)
        self.assertAlmostEqual(result.unemployment, 2.0)
        self.assertAlmostEqual(result.output_growth, result.aggregate_demand)
        self.assertAlmostEqual(result.inflation, result.aggregate_supply)


class HistoryTests(unittest.TestCase):
    def test_economy_builds_demand_intercept_from_rate_history(self):
        parameters = EconomyParameters(
            demand_intercept_weight_10=-0.1,
            demand_intercept_weight_20=-0.1,
        )
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, 1),
            parameters=parameters,
        )
        prior_real_rate = economy.history.entries[-1].real_interest_rate
        prior_equilibrium_rate = economy.history.entries[-1].equilibrium_real_rate
        economy.interest_rate = 4
        economy.simulate_quarter()

        # The only prior real-rate gap feeds both windows, atop potential growth.
        prior_gap = prior_real_rate - prior_equilibrium_rate
        expected_intercept = (
            parameters.potential_growth - 0.1 * prior_gap - 0.1 * prior_gap
        )
        entry = economy.history.entries[-1]
        implied_intercept = entry.gdp_growth - (
            parameters.demand_real_rate
            * (entry.interest_rate - entry.inflation_rate - entry.equilibrium_real_rate)
            + entry.demand_shock
        )
        self.assertAlmostEqual(implied_intercept, expected_intercept)

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
