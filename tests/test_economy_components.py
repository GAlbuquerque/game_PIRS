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
    apply_inflation_floor,
    apply_output_capacity,
    calculate_effective_real_rate,
    calculate_output_growth,
    calculate_real_interest_rate,
    dynamic_is_equation,
    expected_output_gap,
    maximum_output_gap,
    new_keynesian_phillips_curve,
    okuns_law,
    slow_deflation,
    solve_ad_as,
)
from parameters import EconomyParameters
from utils import compute_real_interest_rate


class LawsOfMotionTests(unittest.TestCase):
    def test_default_calibration_matches_model_guide(self):
        parameters = EconomyParameters()
        self.assertEqual(parameters.real_rate_persistence, 0.5)
        self.assertEqual(parameters.output_gap_expectation, 0.8)
        self.assertEqual(parameters.negative_gap_slope_ratio, 0.5)
        self.assertEqual(parameters.deflation_adjustment_ratio, 0.5)
        self.assertEqual(parameters.minimum_unemployment, 1.0)
        self.assertEqual(parameters.minimum_inflation, -99.0)

    def test_real_rate_uses_forward_inflation_expectation(self):
        self.assertEqual(calculate_real_interest_rate(5.0, 2.0), 3.0)

    def test_effective_rate_uses_last_period_real_rate(self):
        parameters = EconomyParameters(real_rate_persistence=0.75)
        self.assertEqual(calculate_effective_real_rate(2.0, 6.0, parameters), 3.0)

    def test_expected_gap_shrinks_the_inherited_gap(self):
        parameters = EconomyParameters(output_gap_expectation=0.8)
        self.assertEqual(expected_output_gap(5.0, parameters), 4.0)

    def test_dynamic_is_uses_effective_rate_gap(self):
        parameters = EconomyParameters(intertemporal_elasticity_inverse=2.0)
        gap = dynamic_is_equation(1.0, 4.0, 2.0, 0.25, parameters)
        self.assertEqual(gap, 0.25)

    def test_output_cap_is_implied_by_one_percent_unemployment(self):
        parameters = EconomyParameters(okun_coefficient=0.5, minimum_unemployment=1.0)
        self.assertEqual(maximum_output_gap(5.0, parameters), 8.0)
        self.assertEqual(apply_output_capacity(20.0, 5.0, parameters), 8.0)
        self.assertEqual(okuns_law(5.0, 8.0, parameters), 1.0)

    def test_negative_gap_halves_phillips_slope(self):
        parameters = EconomyParameters(phillips_output_gap=0.2)
        positive = new_keynesian_phillips_curve(2.0, 2.0, 0.0, parameters)
        negative = new_keynesian_phillips_curve(2.0, -2.0, 0.0, parameters)
        self.assertAlmostEqual(positive, 2.4)
        self.assertAlmostEqual(negative, 1.8)

    def test_negative_phillips_result_is_halved_and_floored(self):
        parameters = EconomyParameters(minimum_inflation=-99.0)
        self.assertEqual(slow_deflation(-2.0, parameters), -1.0)
        self.assertEqual(apply_inflation_floor(-120.0, parameters), -99.0)

    def test_equations_are_applied_sequentially(self):
        parameters = EconomyParameters(
            output_gap_expectation=0.8,
            intertemporal_elasticity_inverse=1.0,
            phillips_output_gap=0.1,
        )
        result = solve_ad_as(
            player_interest_rate=20.0,  # Current i_t must not affect current output.
            equilibrium_real_rate=1.0,
            previous_unemployment=5.0,
            inflation_shock=0.0,
            demand_shock=0.0,
            parameters=parameters,
            previous_inflation=2.0,
            natural_unemployment=5.0,
            previous_output_gap=1.0,
            interest_rate_pressure=2.0,
        )
        self.assertAlmostEqual(result.output_gap, -0.2)
        self.assertAlmostEqual(result.inflation, 1.99)
        self.assertAlmostEqual(result.unemployment, 5.14)
        self.assertAlmostEqual(
            result.output_growth,
            calculate_output_growth(-0.2, 1.0, parameters),
        )

    def test_current_policy_rate_has_no_effect_until_next_period(self):
        arguments = dict(
            equilibrium_real_rate=1.0,
            previous_unemployment=5.0,
            inflation_shock=0.0,
            demand_shock=0.0,
            parameters=EconomyParameters(),
            previous_inflation=2.0,
            interest_rate_pressure=1.0,
        )
        low = solve_ad_as(player_interest_rate=0.0, **arguments)
        high = solve_ad_as(player_interest_rate=20.0, **arguments)
        self.assertEqual(low, high)


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

    def test_economy_updates_effective_rate_from_lagged_real_rate(self):
        parameters = EconomyParameters(real_rate_persistence=0.5)
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, 1),
            parameters=parameters,
        )
        economy.event_engine.events = []
        old_effective_rate = economy.interest_rate_pressure
        lagged_real_rate = economy.history.entries[-1].real_interest_rate

        economy.interest_rate = 20.0
        economy.simulate_quarter()

        expected = 0.5 * old_effective_rate + 0.5 * lagged_real_rate
        self.assertAlmostEqual(economy.interest_rate_pressure, expected)
        # The newly selected rate is recorded, but was not used in R_t yet.
        self.assertAlmostEqual(
            economy.history.entries[-1].real_interest_rate,
            economy.interest_rate - economy.expected_inflation,
        )

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
