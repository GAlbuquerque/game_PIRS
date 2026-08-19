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
    aggregate_demand_curve,
    aggregate_supply_curve,
    calculate_interest_rate_pressure,
    calculate_vertical_supply_output_gap,
    find_curve_intersection,
    solve_ad_as,
)
from parameters import EconomyParameters
from utils import compute_real_interest_rate


class LawsOfMotionTests(unittest.TestCase):
    def test_default_policy_transmission_calibration(self):
        parameters = EconomyParameters()

        self.assertEqual(parameters.phillips_output_gap, 0.1)
        self.assertEqual(parameters.interest_rate_pressure_persistence, 0.5)
        self.assertEqual(parameters.demand_interest_rate_pressure, 1.0)

    def test_interest_rate_pressure_uses_lagged_gap_and_persistence(self):
        parameters = EconomyParameters(
            interest_rate_pressure_persistence=0.5,
        )
        rates = [2.0, 5.0, 9.0]
        equilibrium_rates = [1.0, 2.0, 3.0]

        result = calculate_interest_rate_pressure(
            rates, equilibrium_rates, previous_pressure=1.0, parameters=parameters
        )

        self.assertAlmostEqual(result, 2.0)

    def test_interest_rate_pressure_rejects_mismatched_rate_histories(self):
        with self.assertRaisesRegex(ValueError, "must have equal length"):
            calculate_interest_rate_pressure(
                [1.0, 2.0], [1.0], 0.0, EconomyParameters()
            )

    def test_equilibrium_when_real_rates_always_equal_equilibrium_rates(self):
        parameters = EconomyParameters()
        real_rates = [1.0] * 20
        equilibrium_real_rates = [1.0] * 20
        pressure = calculate_interest_rate_pressure(
            real_rates, equilibrium_real_rates, 0.0, parameters
        )

        # Neutral nominal-demand growth is potential growth plus expected
        # inflation. With a zero inherited gap, neutral policy leaves x at zero.
        result = solve_ad_as(
            player_interest_rate=3.0,
            equilibrium_real_rate=1.0,
            previous_unemployment=5.0,
            inflation_shock=0.0,
            demand_shock=0.0,
            parameters=parameters,
            interest_rate_pressure=pressure,
        )

        self.assertAlmostEqual(pressure, 0.0)
        self.assertAlmostEqual(result.inflation, 2.0)
        self.assertAlmostEqual(result.output_growth, 2.0)
        self.assertAlmostEqual(result.output_gap, 0.0)
        self.assertAlmostEqual(3.0 - result.inflation, 1.0)

    def test_neutral_policy_supports_stable_high_inflation(self):
        parameters = EconomyParameters()
        result = solve_ad_as(
            player_interest_rate=7.0,
            equilibrium_real_rate=1.0,
            previous_unemployment=5.0,
            inflation_shock=0.0,
            demand_shock=0.0,
            parameters=parameters,
            previous_inflation=6.0,
            target_inflation=2.0,
            reputation=0.0,
            natural_unemployment=5.0,
            previous_output_gap=0.0,
            demand_shift=0.0,
        )

        self.assertAlmostEqual(result.inflation, 6.0)
        self.assertAlmostEqual(result.output_gap, 0.0)
        self.assertAlmostEqual(result.output_growth, parameters.potential_growth)
        self.assertAlmostEqual(result.unemployment, 5.0)

    def test_vertical_supply_output_is_derived_from_okuns_law(self):
        parameters = EconomyParameters(
            potential_growth=2.0,
            okun_coefficient=0.4,
            vertical_supply_unemployment=2.0,
        )

        output_gap = calculate_vertical_supply_output_gap(5.0, parameters)

        self.assertAlmostEqual(output_gap, 7.5)
        self.assertAlmostEqual(
            5.0 - parameters.okun_coefficient * output_gap,
            parameters.vertical_supply_unemployment,
        )

    def test_real_interest_rate_uses_linear_approximation(self):
        self.assertEqual(compute_real_interest_rate(10.0, 6.0), 4.0)

    def test_ad_and_as_intersect_and_okun_runs_afterward(self):
        parameters = EconomyParameters()
        result = solve_ad_as(4.0, 1.0, 5.0, 0.1, -0.2, parameters)

        self.assertAlmostEqual(result.output_gap, result.aggregate_demand)
        self.assertAlmostEqual(result.inflation, result.aggregate_supply)
        expected_unemployment = 5.0 - parameters.okun_coefficient * result.output_gap
        self.assertAlmostEqual(result.unemployment, expected_unemployment)

    def test_interest_rate_pressure_reduces_demand(self):
        parameters = EconomyParameters()
        low = solve_ad_as(2, 1, 5, 0, 0, parameters, interest_rate_pressure=-1)
        high = solve_ad_as(6, 1, 5, 0, 0, parameters, interest_rate_pressure=1)
        self.assertGreater(low.output_gap, high.output_gap)

    def test_aggregate_demand_slopes_down_with_inflation(self):
        parameters = EconomyParameters()
        low_inflation = aggregate_demand_curve(2, 4, 1, 0, parameters)
        high_inflation = aggregate_demand_curve(3, 4, 1, 0, parameters)

        self.assertAlmostEqual(high_inflation - low_inflation, -0.25)

    def test_expected_inflation_enters_aggregate_demand_directly(self):
        parameters = EconomyParameters()
        low_expectation = aggregate_demand_curve(
            2, 4, 1, 0, parameters, expected_inflation=2
        )
        high_expectation = aggregate_demand_curve(
            2, 4, 1, 0, parameters, expected_inflation=3
        )

        expected_effect = 1.0 / parameters.periods_per_year
        self.assertAlmostEqual(high_expectation - low_expectation, expected_effect)

    def test_numerical_solution_drives_both_equation_errors_to_zero(self):
        parameters = EconomyParameters()
        result = solve_ad_as(4.0, 1.0, 5.0, 0.1, -0.2, parameters)
        errors = ad_as_errors(
            result.inflation,
            result.output_gap,
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

        # alpha = R / 10 = .08, so beta_0,pi = .08(2) + .92(6) = 5.68.
        self.assertAlmostEqual(result.inflation, 5.68)

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
        parameters = EconomyParameters(demand_intercept=40)
        natural_rate = 5.0
        capacity = (
            natural_rate - parameters.vertical_supply_unemployment
        ) / parameters.okun_coefficient
        result = solve_ad_as(
            0, 0, natural_rate, 0, 0, parameters,
            natural_unemployment=natural_rate,
            vertical_supply_output_gap=capacity,
        )

        self.assertAlmostEqual(result.output_gap, capacity)
        self.assertAlmostEqual(result.unemployment, 2.0)
        self.assertAlmostEqual(result.output_gap, result.aggregate_demand)
        self.assertAlmostEqual(result.inflation, result.aggregate_supply)
        inflation_at_kink = aggregate_supply_curve(capacity, 0, parameters)
        self.assertGreater(result.inflation, inflation_at_kink)

    def test_growth_is_the_annualized_change_in_output_gap(self):
        parameters = EconomyParameters()
        previous_gap = -1.0
        result = solve_ad_as(
            3, 1, 5, 0, 0, parameters,
            previous_output_gap=previous_gap,
        )

        expected_growth = parameters.potential_growth + parameters.periods_per_year * (
            result.output_gap - previous_gap
        )
        self.assertAlmostEqual(result.output_growth, expected_growth)


class HistoryTests(unittest.TestCase):
    def test_history_records_ex_ante_real_rate_and_its_expectation(self):
        economy = Economy(
            initial_state=EconomicIndicators(6, 5, 5, 2, 1),
            parameters=EconomyParameters(event_probability_scale=0),
        )

        entry = economy.history.entries[-1]
        expected_inflation = 0.08 * 2 + 0.92 * 6
        self.assertAlmostEqual(entry.expected_inflation, expected_inflation)
        self.assertAlmostEqual(
            entry.real_interest_rate,
            entry.interest_rate - expected_inflation,
        )
        self.assertNotAlmostEqual(
            entry.real_interest_rate,
            entry.interest_rate - entry.inflation_rate,
        )

    def test_background_equilibrium_rate_mean_reverts_and_includes_shock(self):
        parameters = EconomyParameters(
            equilibrium_real_rate_anchor=0.5,
            equilibrium_real_rate_reversion=0.02,
        )
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, 2.5),
            parameters=parameters,
        )

        economy._apply_background_shocks(np.array([0.0, 0.0, 0.0, 0.1]))

        expected_rate = 2.5 - 0.02 * (2.5 - 0.5) + 0.1
        self.assertAlmostEqual(economy.indicators.real_rate_eq, expected_rate)

    def test_background_equilibrium_rate_reverts_up_when_below_anchor(self):
        parameters = EconomyParameters(
            equilibrium_real_rate_anchor=0.5,
            equilibrium_real_rate_reversion=0.02,
        )
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, -0.5),
            parameters=parameters,
        )

        economy._apply_background_shocks(np.zeros(4))

        self.assertAlmostEqual(economy.indicators.real_rate_eq, -0.48)

    def test_economy_builds_interest_rate_pressure_from_rate_history(self):
        parameters = EconomyParameters(
            interest_rate_pressure_persistence=0.5,
        )
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, 1),
            parameters=parameters,
        )
        economy.event_engine.events = []
        prior_real_rate = economy.history.entries[-1].real_interest_rate
        prior_equilibrium_rate = economy.history.entries[-1].equilibrium_real_rate
        economy.interest_rate = 4
        economy.simulate_quarter()
        economy.simulate_quarter()

        # Quarter two uses the initial state's real-rate gap (t-2).
        prior_gap = prior_real_rate - prior_equilibrium_rate
        entry = economy.history.entries[-1]
        self.assertAlmostEqual(entry.interest_rate_pressure, 0.5 * prior_gap)

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
        self.assertEqual(len(history.series("expected_inflation")), 6)
        for entry in history.entries:
            self.assertAlmostEqual(
                entry.real_interest_rate,
                entry.interest_rate - entry.expected_inflation,
            )
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


class PersonaReactionTests(unittest.TestCase):
    def _indicators(self, inflation=2.0, unemployment=5.0):
        return EconomicIndicators(
            inflation_rate=inflation,
            unemployment_rate=unemployment,
            natural_unemployment_rate=5.0,
            real_rate_eq=1.0,
            gdp_growth=2.0,
            target_inflation_rate=2.0,
        )

    def test_automated_rate_moves_gradually_toward_distant_rule_rate(self):
        from personas import automated_rate

        self.assertEqual(automated_rate("hawk", 2.0, self._indicators()), 2.5)

    def test_automated_rate_holds_within_half_point_deadband(self):
        from personas import automated_rate

        self.assertEqual(automated_rate("good", 3.2, self._indicators()), 3.25)

    def test_automated_rate_uses_emergency_recession_cut(self):
        from personas import automated_rate

        indicators = self._indicators(inflation=0.5, unemployment=7.0)
        self.assertEqual(automated_rate("hawk", 4.0, indicators), 0.0)


if __name__ == "__main__":
    unittest.main()
