"""Focused tests for the editable economy components."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

from history import EconomicHistory
from economy import Economy
from event_engine import EventEngine
from events import GameEvent, initialize_events
from indicators import EconomicIndicators
from laws_of_motion import (
    ModelResult,
    apply_output_capacity,
    calculate_expected_inflation,
    calculate_expected_output_gap,
    calculate_interest_rate_pressure,
    calculate_minimum_output_gap,
    calculate_maximum_output_gap,
    calculate_real_interest_rate,
    calculate_real_interest_rate_gap,
    dynamic_is_equation,
    new_keynesian_phillips_curve,
    okuns_law,
    phillips_curve_gap_effect,
    calculate_quarter_outcome,
)
from parameters import EconomyParameters
from endgame_logic import (
    EndGameContext,
    _performance_band,
    build_end_of_term_message,
    classify_public_view,
    evaluate_end_of_term,
    mandate_loss,
    mandate_targets,
    taylor_policy_deviations,
)


class LawsOfMotionTests(unittest.TestCase):
    def test_default_calibration_matches_documented_model(self):
        parameters = EconomyParameters()

        self.assertEqual(parameters.output_gap_expectation_persistence, 0.8)
        self.assertEqual(parameters.interest_rate_pressure_persistence, 0.8)
        self.assertEqual(parameters.intertemporal_elasticity_inverse, 2.0)
        self.assertEqual(parameters.phillips_output_gap, 0.1)
        self.assertEqual(parameters.deflation_adjustment_ratio, 0.8)
        self.assertEqual(parameters.reputation_expectation_coefficient, 0.2)
        self.assertEqual(parameters.okun_coefficient, 0.7)
        self.assertEqual(parameters.equilibrium_real_rate_reversion, 0.02)
        self.assertEqual(parameters.shock_std_devs, (0.3, 0.285714, 0.05, 0.0))
        np.testing.assert_array_equal(
            parameters.shock_correlations, np.eye(4)
        )
        self.assertEqual(parameters.minimum_unemployment, 1.0)
        self.assertEqual(parameters.maximum_unemployment, 70.0)
        self.assertEqual(parameters.minimum_inflation, -99.0)

    def test_configured_interest_rate_floor_is_enforced(self):
        economy = Economy(minimum_interest_rate=-0.75)
        economy.adjust_interest_rate(-1.0)
        self.assertEqual(economy.interest_rate, -0.75)
        economy.adjust_interest_rate(-0.5)
        self.assertEqual(economy.interest_rate, -0.5)

    def test_player_events_are_restricted_to_central_banker_mode(self):
        economy = Economy(difficulty="senior")
        succeeded, reason = economy.trigger_player_event("quantitative_easing")
        self.assertFalse(succeeded)
        self.assertIn("Central Banker", reason)

    def test_only_one_player_event_can_be_triggered_each_quarter(self):
        economy = Economy(difficulty="central_banker")
        self.assertTrue(economy.trigger_player_event("quantitative_easing")[0])
        succeeded, reason = economy.trigger_player_event("low_rate_guidance")
        self.assertFalse(succeeded)
        self.assertIn("Only one", reason)

    def test_high_rate_guidance_requires_reputation_above_point_seven(self):
        economy = Economy(difficulty="central_banker")
        economy.reputation = 0.7
        self.assertFalse(economy.trigger_player_event("high_rate_guidance")[0])
        economy.reputation = 0.71
        self.assertTrue(economy.trigger_player_event("high_rate_guidance")[0])

    def test_qe_peaks_next_quarter_then_dissipates(self):
        economy = Economy(difficulty="central_banker")
        economy.trigger_player_event("quantitative_easing")
        immediate = economy._current_player_event_effects()
        economy.current_quarter += 1
        peak = economy._current_player_event_effects()
        economy.current_quarter += 1
        decay = economy._current_player_event_effects()
        self.assertGreater(peak["demand"], immediate["demand"])
        self.assertGreater(peak["inflation"], immediate["inflation"])
        self.assertLess(decay["demand"], peak["demand"])
        self.assertLess(decay["inflation"], peak["inflation"])

    def test_qe_is_active_for_its_eight_quarter_schedule(self):
        economy = Economy(difficulty="central_banker")
        economy.trigger_player_event("quantitative_easing")
        for _ in range(8):
            self.assertTrue(economy._player_event_is_active("quantitative_easing"))
            economy.current_quarter += 1
        self.assertFalse(economy._player_event_is_active("quantitative_easing"))

    def test_forward_guidance_lasts_four_quarters_and_has_cooldown(self):
        economy = Economy(difficulty="central_banker")
        self.assertTrue(economy.trigger_player_event("low_rate_guidance")[0])
        for expected_age in range(4):
            self.assertEqual(
                economy._current_player_event_effects()["rate_pressure"], -1.0
            )
            if expected_age < 3:
                economy.current_quarter += 1
                economy.player_event_used_quarter = None
                self.assertFalse(economy.trigger_player_event("low_rate_guidance")[0])
        economy.current_quarter += 1
        economy.player_event_used_quarter = None
        self.assertEqual(economy._current_player_event_effects()["rate_pressure"], 0.0)
        self.assertTrue(economy.trigger_player_event("low_rate_guidance")[0])

    def test_mandate_targets_use_configured_values(self):
        self.assertEqual(
            mandate_targets("dual_mandate", 6.5, 3.0),
            {"inflation": 3.0, "unemployment": 6.5},
        )
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=3.0,
            initial_unemployment=5.0,
            dual_unemployment_target=5.0,
            inflation_history=[3.0] * 12,
            unemployment_history=[5.0] * 12,
            real_interest_rate_history=[1.0] * 12,
            inflation_target=3.0,
        )
        self.assertIn("You kept inflation close to target", build_end_of_term_message(context))

    def test_inflation_mandate_loss_ignores_unemployment(self):
        calm = mandate_loss("inflation_target", [2.0] * 16, [4.0] * 16, 2.0, 4.0)
        depression = mandate_loss(
            "inflation_target", [2.0] * 16, [30.0] * 16, 2.0, 4.0
        )
        self.assertEqual(calm, 0.0)
        self.assertEqual(depression, calm)

    def test_mandate_loss_penalizes_inflation_volatility(self):
        loss = mandate_loss(
            "inflation_target", [-3.0, 7.0] * 8, [4.0] * 16, 2.0, 4.0
        )
        self.assertEqual(loss, 5.0)

    def test_dual_mandate_loss_combines_inflation_and_unemployment(self):
        loss = mandate_loss("dual_mandate", [3.0] * 16, [6.0] * 16, 2.0, 4.0)
        self.assertAlmostEqual(loss, 1.5)

    def test_taylor_deviation_uses_feasible_rate_at_lower_bound(self):
        deviations, constrained = taylor_policy_deviations(
            inflation_history=[-1.0],
            unemployment_history=[10.0],
            natural_unemployment_history=[4.0],
            equilibrium_real_rate_history=[1.0],
            selected_rate_history=[0.0],
            inflation_target=2.0,
            minimum_interest_rate=0.0,
        )
        self.assertEqual(deviations, [0.0])
        self.assertEqual(constrained, 1)

    def test_public_view_uses_doubled_taylor_deviation_thresholds(self):
        self.assertEqual(classify_public_view([1.0] * 16)[0], "Balanced")
        self.assertEqual(classify_public_view([1.01] * 16)[0], "Hawk")
        self.assertEqual(classify_public_view([-1.0] * 16)[0], "Balanced")
        self.assertEqual(classify_public_view([-1.01] * 16)[0], "Dove")
        self.assertEqual(classify_public_view([-4.0] * 16)[0], "Dove")
        self.assertEqual(classify_public_view([-4.01] * 16)[0], "Careless")

    def test_public_view_classifies_large_unbiased_swings_as_erratic(self):
        label, message = classify_public_view([-5.0, 5.0] * 8)

        self.assertEqual(label, "Erratic")
        self.assertEqual(
            message,
            "Your decisions repeatedly swung between unusually tight and unusually "
            "loose policy, leaving markets unable to discern a stable strategy.",
        )

    def test_erratic_classification_requires_strict_sigma_threshold(self):
        self.assertEqual(classify_public_view([-4.0, 4.0] * 8)[0], "Balanced")
        self.assertEqual(classify_public_view([11.0] * 16)[0], "Hawk")

    def test_message_combines_context_record_direction_and_original_reputation(self):
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=8.0,
            initial_unemployment=5.0,
            dual_unemployment_target=4.0,
            inflation_history=[5.0] * 4 + [3.0] * 8 + [2.0] * 4,
            unemployment_history=[5.0] * 16,
            real_interest_rate_history=[1.0] * 16,
            inflation_target=2.0,
            term_event_names=["Global Supply Shock"],
            policy_deviation_history=[1.5] * 16,
        )
        message = build_end_of_term_message(context)
        self.assertIn(
            "Your term has ended. It was marked by a global supply shock.",
            message,
        )
        self.assertIn("Despite those shocks, your results were strong.", message)
        self.assertIn(
            "Bond markets saw you as inflation-first and uncompromising.", message
        )
        self.assertIn("They classify you as: Hawk", message)
        self.assertIn("You kept inflation close to target", message)
        self.assertIn("By the final year, the economy stood closer", message)

    def test_performance_bands_use_term_and_ending_losses(self):
        strong = EndGameContext(
            mandate="inflation_target",
            initial_inflation=2.0,
            initial_unemployment=4.0,
            dual_unemployment_target=4.0,
            inflation_history=[2.5] * 16,
            unemployment_history=[4.0] * 16,
            real_interest_rate_history=[1.0] * 16,
        )
        strong_despite_weak_beginning = EndGameContext(
            **{
                **strong.__dict__,
                "inflation_history": [3.5] * 4 + [2.0] * 12,
            }
        )
        mixed_from_weak_ending = EndGameContext(
            **{
                **strong.__dict__,
                "inflation_history": [2.0] * 12 + [3.5] * 4,
            }
        )
        poor = EndGameContext(
            **{**strong.__dict__, "inflation_history": [7.0] * 16}
        )
        upper_mixed_boundary = EndGameContext(
            **{**strong.__dict__, "inflation_history": [6.0] * 16}
        )
        self.assertEqual(evaluate_end_of_term(strong)["performance"], "strong")
        self.assertEqual(
            evaluate_end_of_term(strong_despite_weak_beginning)["performance"],
            "strong",
        )
        self.assertEqual(
            evaluate_end_of_term(mixed_from_weak_ending)["performance"],
            "mixed",
        )
        self.assertEqual(evaluate_end_of_term(poor)["performance"], "poor")
        self.assertEqual(
            evaluate_end_of_term(upper_mixed_boundary)["performance"], "mixed"
        )

    def test_dual_mandate_uses_its_own_strong_boundaries(self):
        self.assertEqual(_performance_band("dual_mandate", 2.99, 1.99), "strong")
        self.assertEqual(_performance_band("dual_mandate", 3.0, 1.99), "mixed")
        self.assertEqual(_performance_band("dual_mandate", 2.99, 2.0), "mixed")
        self.assertEqual(_performance_band("dual_mandate", 6.0, 3.0), "mixed")
        self.assertEqual(_performance_band("dual_mandate", 6.01, 3.0), "poor")
        self.assertEqual(_performance_band("inflation_target", 4.01, 1.01), "poor")
        self.assertEqual(_performance_band("inflation_target", 2.01, 1.01), "mixed")

    def test_below_target_failure_mentions_deflation_not_price_stability(self):
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=-3.0,
            initial_unemployment=4.0,
            dual_unemployment_target=4.0,
            inflation_history=[-3.0] * 16,
            unemployment_history=[4.0] * 16,
            real_interest_rate_history=[1.0] * 16,
            inflation_target=2.0,
        )
        message = build_end_of_term_message(context)
        self.assertIn("Inflation fell well below target", message)
        self.assertIn("deflationary pressure", message)

    def test_favorable_event_aids_a_strong_performance(self):
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=2.0,
            initial_unemployment=4.0,
            dual_unemployment_target=4.0,
            inflation_history=[2.5] * 16,
            unemployment_history=[4.0] * 16,
            real_interest_rate_history=[1.0] * 16,
            term_event_names=["Technological Boom"],
        )
        self.assertIn(
            "It was marked by a technological boom.",
            build_end_of_term_message(context),
        )
        self.assertIn(
            "Aided by those favorable conditions, your results were strong.",
            build_end_of_term_message(context),
        )

    def test_quiet_term_is_credited_and_adversity_only_uses_despite_for_success(self):
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=2.0,
            initial_unemployment=4.0,
            dual_unemployment_target=4.0,
            inflation_history=[2.5] * 16,
            unemployment_history=[4.0] * 16,
            real_interest_rate_history=[1.0] * 16,
        )
        self.assertIn(
            "The term passed without a major economic shock.",
            build_end_of_term_message(context),
        )
        self.assertIn(
            "Helped by that calm backdrop, your results were strong.",
            build_end_of_term_message(context),
        )
        context.term_event_names = ["Financial Crisis"]
        self.assertIn(
            "It was marked by a financial crisis.",
            build_end_of_term_message(context),
        )
        self.assertIn(
            "Despite those shocks, your results were strong.",
            build_end_of_term_message(context),
        )

    def test_crises_are_adverse_and_favorable_events_use_despite_on_failure(self):
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=7.0,
            initial_unemployment=4.0,
            dual_unemployment_target=4.0,
            inflation_history=[7.0] * 16,
            unemployment_history=[4.0] * 16,
            real_interest_rate_history=[1.0] * 16,
            term_event_names=["Financial Crisis", "Major Financial Crisis"],
        )
        self.assertIn(
            "It was marked by a major financial crisis.",
            build_end_of_term_message(context),
        )
        self.assertNotIn("a financial crisis and", build_end_of_term_message(context))
        self.assertIn(
            "In this difficult scenario, your results were poor.",
            build_end_of_term_message(context),
        )
        context.term_event_names = ["Technological Boom"]
        self.assertIn(
            "Despite those favorable conditions, your results were poor.",
            build_end_of_term_message(context),
        )

    def test_spending_wave_alias_and_dominance_read_naturally(self):
        context = EndGameContext(
            mandate="inflation_target",
            initial_inflation=7.0,
            initial_unemployment=4.0,
            dual_unemployment_target=4.0,
            inflation_history=[7.0] * 16,
            unemployment_history=[4.0] * 16,
            real_interest_rate_history=[1.0] * 16,
            term_event_names=["Fiscal Deficit", "Spending Wave"],
        )
        message = build_end_of_term_message(context)
        self.assertIn("It was marked by a spending wave.", message)
        self.assertNotIn("fiscal deficit", message)

    def test_inflation_expectation_uses_reputation_times_anchoring_strength(self):
        parameters = EconomyParameters(reputation_expectation_coefficient=0.5)
        expectation = calculate_expected_inflation(6, 2, 0.8, parameters)
        self.assertAlmostEqual(expectation, 0.4 * 2 + 0.6 * 6)

    def test_fallback_inflation_expectation_is_always_the_target(self):
        parameters = EconomyParameters(
            inflation_target=3.0, expected_inflation=99.0
        )
        self.assertEqual(
            calculate_expected_inflation(None, None, 0.8, parameters), 3.0
        )

    def test_inflation_expectation_rejects_values_outside_unit_interval(self):
        parameters = EconomyParameters(reputation_expectation_coefficient=1.1)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_expected_inflation(6, 2, 0.8, parameters)

    def test_expected_output_gap_shrinks_the_observed_gap(self):
        parameters = EconomyParameters(output_gap_expectation_persistence=0.75)
        self.assertAlmostEqual(calculate_expected_output_gap(4.0, parameters), 3.0)

    def test_expected_output_gap_rejects_invalid_persistence(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_expected_output_gap(
                2.0, EconomyParameters(output_gap_expectation_persistence=1.1)
            )

    def test_real_rate_and_rate_gap_are_separate_equations(self):
        real_rate = calculate_real_interest_rate(5.0, 2.5)
        rate_gap = calculate_real_interest_rate_gap(real_rate, 1.0)
        self.assertAlmostEqual(real_rate, 2.5)
        self.assertAlmostEqual(rate_gap, 1.5)

    def test_interest_pressure_uses_t_minus_one_gap_and_documented_weights(self):
        parameters = EconomyParameters(interest_rate_pressure_persistence=0.75)
        rates = [2.0, 5.0, 9.0]
        natural_rates = [1.0, 2.0, 3.0]

        result = calculate_interest_rate_pressure(
            rates, natural_rates, previous_pressure=2.0, parameters=parameters
        )

        # R_t = .75(2) + .25(9-3) = 3.
        self.assertAlmostEqual(result, 3.0)

    def test_interest_pressure_with_no_history_carries_previous_stock(self):
        result = calculate_interest_rate_pressure(
            [], [], previous_pressure=1.25, parameters=EconomyParameters()
        )
        self.assertAlmostEqual(result, 1.25)

    def test_interest_pressure_rejects_mismatched_histories(self):
        with self.assertRaisesRegex(ValueError, "must have equal length"):
            calculate_interest_rate_pressure(
                [1.0, 2.0], [1.0], 0.0, EconomyParameters()
            )

    def test_dynamic_is_equation_is_calculated_directly(self):
        parameters = EconomyParameters(
            intertemporal_elasticity_inverse=2.0,
        )
        result = dynamic_is_equation(
            expected_future_output_gap=1.0,
            effective_real_rate_gap=2.0,
            demand_shock=0.25,
            parameters=parameters,
        )
        self.assertAlmostEqual(result, 0.25)

    def test_effective_past_tightening_reduces_current_output(self):
        parameters = EconomyParameters()
        easy = calculate_quarter_outcome(5, 0, 0, parameters, interest_rate_pressure=-1)
        tight = calculate_quarter_outcome(5, 0, 0, parameters, interest_rate_pressure=1)
        self.assertGreater(easy.output_gap, tight.output_gap)

    def test_capacity_is_derived_from_one_percent_unemployment(self):
        parameters = EconomyParameters(okun_coefficient=0.5, minimum_unemployment=1.0)
        capacity = calculate_maximum_output_gap(5.0, parameters)
        self.assertAlmostEqual(capacity, 8.0)
        self.assertAlmostEqual(okuns_law(5.0, capacity, parameters), 1.0)

    def test_capacity_clips_positive_output_and_extreme_recessions(self):
        self.assertEqual(apply_output_capacity(10.0, 6.0, -8.0), 6.0)
        self.assertEqual(apply_output_capacity(-10.0, 6.0, -8.0), -8.0)

    def test_minimum_gap_is_derived_from_seventy_percent_unemployment(self):
        parameters = EconomyParameters(okun_coefficient=0.5, maximum_unemployment=70.0)
        floor = calculate_minimum_output_gap(5.0, parameters)
        self.assertAlmostEqual(floor, -130.0)
        self.assertAlmostEqual(okuns_law(5.0, floor, parameters), 70.0)

    def test_phillips_slope_is_symmetric(self):
        parameters = EconomyParameters(phillips_output_gap=0.4)
        self.assertAlmostEqual(phillips_curve_gap_effect(2, parameters), 0.8)
        self.assertAlmostEqual(phillips_curve_gap_effect(-2, parameters), -0.8)

    def test_phillips_curve_is_its_own_equation(self):
        parameters = EconomyParameters(
            inflation_expectation_discount=0.9,
            phillips_output_gap=0.4,
        )
        inflation = new_keynesian_phillips_curve(2, 1.5, 0.1, parameters)
        self.assertAlmostEqual(inflation, 2.5)

    def test_resulting_deflation_is_slowed_by_d(self):
        parameters = EconomyParameters(deflation_adjustment_ratio=0.5)
        deflation = calculate_quarter_outcome(
            5, -2, 0, parameters, previous_inflation=0, reputation=0
        )
        inflation = calculate_quarter_outcome(
            5, 2, 0, parameters, previous_inflation=0, reputation=0
        )
        self.assertAlmostEqual(deflation.inflation, -1.0)
        self.assertAlmostEqual(inflation.inflation, 2.0)

    def test_deflation_floor_is_applied_after_slowdown(self):
        parameters = EconomyParameters(
            deflation_adjustment_ratio=0.5, minimum_inflation=-99.0
        )
        result = calculate_quarter_outcome(
            5, -300, 0, parameters, previous_inflation=0, reputation=0
        )
        self.assertAlmostEqual(result.inflation, -99.0)

    def test_okun_uses_the_gap_version(self):
        parameters = EconomyParameters(okun_coefficient=0.5)
        self.assertAlmostEqual(okuns_law(5.0, 2.0, parameters), 4.0)
        self.assertAlmostEqual(okuns_law(5.0, -2.0, parameters), 6.0)

    def test_neutral_steady_state_is_preserved(self):
        parameters = EconomyParameters()
        result = calculate_quarter_outcome(
            natural_unemployment=5.0,
            inflation_shock=0.0,
            demand_shock=0.0,
            parameters=parameters,
            previous_inflation=6.0,
            target_inflation=2.0,
            reputation=0.0,
            previous_output_gap=0.0,
            interest_rate_pressure=0.0,
        )
        self.assertAlmostEqual(result.inflation, 6.0)
        self.assertAlmostEqual(result.output_gap, 0.0)
        self.assertAlmostEqual(result.unemployment, 5.0)

    def test_integrated_solution_applies_capacity_before_phillips_curve(self):
        parameters = EconomyParameters(
            output_gap_expectation_persistence=0,
            phillips_output_gap=0.2,
            minimum_unemployment=1,
            okun_coefficient=0.5,
        )
        result = calculate_quarter_outcome(
            5, 0, 100, parameters,
            previous_inflation=2,
            reputation=0,
        )
        capacity = (5 - 1) / 0.5
        self.assertAlmostEqual(result.output_gap, capacity)
        self.assertAlmostEqual(result.unemployment, 1.0)
        self.assertAlmostEqual(result.inflation, 2 + 0.2 * capacity)

    def test_integrated_solution_caps_unemployment_at_seventy_percent(self):
        parameters = EconomyParameters(
            output_gap_expectation_persistence=0,
            phillips_output_gap=0.2,
            maximum_unemployment=70,
            okun_coefficient=0.5,
            minimum_inflation=-99,
        )
        result = calculate_quarter_outcome(
            5, 0, -1_000, parameters,
            previous_inflation=2,
            reputation=0,
        )
        floor = (5 - 70) / 0.5
        self.assertAlmostEqual(result.output_gap, floor)
        self.assertAlmostEqual(result.unemployment, 70.0)
        self.assertAlmostEqual(result.inflation, (2 + 0.2 * floor) * 0.8)


class HistoryTests(unittest.TestCase):
    def test_history_records_ex_ante_real_rate_and_its_expectation(self):
        economy = Economy(
            initial_state=EconomicIndicators(6, 5, 5, 2, 1),
            parameters=EconomyParameters(event_probability_scale=0),
        )

        entry = economy.history.entries[-1]
        expected_inflation = 0.16 * 2 + 0.84 * 6
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
        quarter_one = economy.history.entries[-1]
        expected_quarter_one_pressure = 0.5 * (
            prior_real_rate - prior_equilibrium_rate
        )
        self.assertAlmostEqual(
            quarter_one.interest_rate_pressure, expected_quarter_one_pressure
        )

        economy.simulate_quarter()

        # Quarter two uses the rate gap recorded in quarter one (t-1), not the
        # newly selected rate from within quarter two.
        quarter_one_gap = (
            quarter_one.real_interest_rate - quarter_one.equilibrium_real_rate
        )
        entry = economy.history.entries[-1]
        expected_quarter_two_pressure = (
            0.5 * expected_quarter_one_pressure + 0.5 * quarter_one_gap
        )
        self.assertAlmostEqual(
            entry.interest_rate_pressure, expected_quarter_two_pressure
        )

    def test_economy_uses_configured_minimum_inflation(self):
        parameters = EconomyParameters(minimum_inflation=-20.0)
        economy = Economy(
            initial_state=EconomicIndicators(2, 5, 5, 2, 1),
            parameters=parameters,
        )
        motion = ModelResult(-30.0, 5.0, 0.0)

        economy._commit_motion(motion, previous_inflation=2.0)

        self.assertEqual(economy.indicators.inflation_rate, -20.0)

    def test_random_history_retains_all_relevant_series(self):
        initial = EconomicIndicators(2, 5, 5, 2, 1)
        history = EconomicHistory.generate_random(
            6, initial, EconomyParameters(), np.random.default_rng(7)
        )
        self.assertEqual(len(history.entries), 6)
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
    def test_event_unemployment_effects_are_converted_to_output_with_okun(self):
        financial_crisis = next(
            event for event in initialize_events(okun_coefficient=0.5)
            if event.name == "Financial Crisis"
        )

        self.assertNotIn("unemployment", financial_crisis.effects_schedule)
        self.assertEqual(
            financial_crisis.effects_schedule["demand"],
            [-0.2, -2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            financial_crisis.effects_schedule["natural_unemployment"],
            [0.2, 0.2, 0.5, 0, 0, -0.2, -0.2, -0.5],
        )

    def test_all_initialized_events_avoid_direct_unemployment_effects(self):
        for event in initialize_events():
            with self.subTest(event=event.name):
                self.assertNotIn("unemployment", event.effects_schedule)

    def test_recent_qe_halves_major_crisis_escalation_risk(self):
        major_crisis = next(
            event for event in initialize_events()
            if event.name == "Major Financial Crisis"
        )
        recent_crisis = {"past_events": [["Financial Crisis"]]}

        for interest_rate, inflation, expected, mitigated in (
            (5.0, 2.0, 0.1025, 0.05125),
            (1.0, 2.0, 0.0525, 0.02625),
        ):
            history = {
                **recent_crisis,
                "interest_rate": [interest_rate],
                "inflation_rate": [inflation],
            }
            self.assertAlmostEqual(major_crisis.get_probability(history), expected)
            history["recent_quantitative_easing"] = True
            self.assertAlmostEqual(major_crisis.get_probability(history), mitigated)

    def test_qe_alone_does_not_reduce_major_crisis_baseline_risk(self):
        major_crisis = next(
            event for event in initialize_events()
            if event.name == "Major Financial Crisis"
        )
        history = {
            "past_events": [[]],
            "interest_rate": [5.0],
            "inflation_rate": [2.0],
            "recent_quantitative_easing": True,
        }
        self.assertAlmostEqual(major_crisis.get_probability(history), 0.0025)

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
            target_inflation_rate=2.0,
        )

    def test_automated_rate_raises_by_half_point_for_gap_above_one(self):
        from personas import automated_rate

        self.assertEqual(automated_rate("hawk", 2.0, self._indicators()), 2.5)

    def test_automated_rate_cuts_by_half_point_for_gap_below_negative_one(self):
        from personas import automated_rate

        self.assertEqual(automated_rate("good", 5.0, self._indicators()), 4.5)

    def test_automated_rate_holds_within_half_point_deadband(self):
        from personas import automated_rate

        self.assertEqual(automated_rate("good", 3.2, self._indicators()), 3.25)

    def test_hawk_does_not_use_emergency_recession_cut(self):
        from personas import automated_rate

        indicators = self._indicators(inflation=0.5, unemployment=7.0)
        self.assertEqual(automated_rate("hawk", 4.0, indicators), 3.5)

    def test_other_personas_retain_emergency_recession_cut(self):
        from personas import automated_rate

        indicators = self._indicators(inflation=0.5, unemployment=7.0)
        for persona in ("good", "dove", "careless"):
            with self.subTest(persona=persona):
                self.assertEqual(automated_rate(persona, 4.0, indicators), 0.0)

    def test_dove_and_careless_do_not_use_emergency_inflation_hike(self):
        from personas import automated_rate

        indicators = self._indicators(inflation=20.0, unemployment=5.0)
        for persona in ("dove", "careless"):
            with self.subTest(persona=persona):
                self.assertEqual(automated_rate(persona, 2.0, indicators), 2.5)

    def test_good_and_hawk_retain_emergency_inflation_hike(self):
        from personas import automated_rate

        indicators = self._indicators(inflation=20.0, unemployment=5.0)
        for persona in ("good", "hawk"):
            with self.subTest(persona=persona):
                self.assertEqual(automated_rate(persona, 2.0, indicators), 28.0)


if __name__ == "__main__":
    unittest.main()
