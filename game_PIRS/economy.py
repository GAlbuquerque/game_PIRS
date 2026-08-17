#!/usr/bin/env python3
"""High-level coordinator for one quarter of the economic simulation."""

from dataclasses import replace

import numpy as np

from event_engine import EventEngine
from history import EconomicHistory
from indicators import EconomicIndicators
from laws_of_motion import (
    calculate_demand_shift,
    calculate_vertical_supply_output_gap,
    solve_ad_as,
)
from parameters import EconomyParameters
from personas import automated_rate, draw_persona
from reputation import update_reputation
from shocks import generate_shocks
from utils import compute_real_interest_rate
from variables import Variables


class Economy:
    """Coordinate events, shocks, laws of motion, reputation, and history."""

    EVENT_HORIZON = 8

    def __init__(
        self,
        initial_state=None,
        difficulty="central_banker",
        scenario=None,
        parameters=None,
        random_history_quarters=0,
    ):
        self.parameters = parameters or EconomyParameters()
        self.difficulty = difficulty
        self.shock_sd_scale = self._difficulty_shock_scale(difficulty)
        self.indicators = initial_state or EconomicIndicators.generate_random_initial_state()
        if scenario is not None:
            self.indicators = replace(self.indicators, **scenario)
        self.indicators.potential_growth = self.parameters.potential_growth
        if self.indicators.output_gap is None:
            self.indicators.output_gap = (
                self.indicators.natural_unemployment_rate
                - self.indicators.unemployment_rate
            ) / self.parameters.okun_coefficient

        self.interest_rate = max(float(np.random.normal(0.5, 2)), 0.0)
        self.reputation = 0.8
        self.cb_persona = draw_persona()
        self.current_quarter = 1
        self.max_quarters = 50
        self.offset = 0
        self.player_start_turn = 40

        self.event_engine = EventEngine(
            difficulty=difficulty,
            horizon=self.EVENT_HORIZON,
            cooldown_quarters=self._difficulty_event_cooldown(difficulty),
        )
        self.history = EconomicHistory.generate_random(
            random_history_quarters, self.indicators, self.parameters
        )
        # This is a structural capacity limit, fixed once at turn 1 rather than
        # moving with later natural-rate shocks.
        self._vertical_as_output_gap = calculate_vertical_supply_output_gap(
            self.indicators.natural_unemployment_rate,
            self.parameters,
        )
        self.variables = Variables()  # Compatibility view consumed by the existing UI.
        self._record_initial_state()

    @staticmethod
    def _difficulty_event_cooldown(difficulty):
        return {"principles": 20, "senior": 10, "central_banker": 0}.get(
            difficulty, 0
        )

    @staticmethod
    def _difficulty_shock_scale(difficulty):
        return {"principles": 0.1, "senior": 0.5, "central_banker": 1.0}.get(
            difficulty, 1.0
        )

    def simulate_quarter(self):
        """Advance exactly one quarter and return its event and shock summary."""
        event_history = self.history.event_snapshot(
            self.current_quarter - self.offset,
            self.event_engine.past_events,
        )
        outcome = self.event_engine.advance(
            event_history, self.current_quarter, self.player_start_turn
        )
        event_effects = dict(outcome.effects)
        event_inflation = event_effects.pop("inflation", 0.0)
        self.apply_event_effects(event_effects)

        shocks = generate_shocks(
            self.parameters.shock_correlations,
            self.parameters.std_devs * self.shock_sd_scale,
        )
        previous_inflation = self.indicators.inflation_rate
        self._apply_background_shocks(shocks)
        real_rates = self.history.series("real_interest_rate")
        equilibrium_real_rates = self.history.series("equilibrium_real_rate")
        demand_shift = calculate_demand_shift(
            real_rates, equilibrium_real_rates, self.parameters
        )
        motion = solve_ad_as(
            player_interest_rate=self.interest_rate,
            equilibrium_real_rate=self.indicators.real_rate_eq,
            previous_unemployment=self.indicators.unemployment_rate,
            inflation_shock=shocks[0] + event_inflation,
            demand_shock=shocks[1],
            parameters=self.parameters,
            previous_inflation=previous_inflation,
            target_inflation=self.indicators.target_inflation_rate,
            reputation=self.reputation,
            natural_unemployment=self.indicators.natural_unemployment_rate,
            previous_output_gap=self.indicators.output_gap,
            demand_shift=demand_shift,
            vertical_supply_output_gap=self._vertical_as_output_gap,
        )
        self._commit_motion(motion, previous_inflation)
        recorded_shocks = shocks.copy()
        recorded_shocks[0] += event_inflation
        self._record_quarter(motion, recorded_shocks, outcome.name)
        self.current_quarter += 1
        return {
            "event": outcome.description,
            "event_name": outcome.name,
            "gap_effect": motion.output_gap,  # Legacy result key used by the UI.
            "shocks": shocks.tolist(),
        }

    def set_difficulty(self, difficulty):
        """Update difficulty-dependent shock and event settings together."""
        self.difficulty = difficulty
        self.shock_sd_scale = self._difficulty_shock_scale(difficulty)
        self.event_engine.difficulty = difficulty
        self.event_engine.cooldown_quarters = self._difficulty_event_cooldown(
            difficulty
        )

    def event_history(self):
        """Build the current history view used to evaluate event probabilities."""
        return self.history.event_snapshot(
            self.current_quarter - self.offset, self.event_engine.past_events
        )

    def _apply_background_shocks(self, shocks):
        """Evolve natural unemployment and r* before solving this quarter's AD-AS."""
        p = self.parameters
        natural_drift = -p.natural_unemployment_reversion * (
            self.indicators.natural_unemployment_rate
            - p.natural_unemployment_anchor
        )
        self.indicators.natural_unemployment_rate = max(
            p.minimum_natural_unemployment,
            self.indicators.natural_unemployment_rate + natural_drift + shocks[2],
        )
        self.indicators.real_rate_eq += shocks[3]

    def _commit_motion(self, motion, previous_inflation):
        self.indicators.inflation_rate = max(
            float(motion.inflation), self.parameters.minimum_inflation
        )
        self.indicators.gdp_growth = float(motion.output_growth)
        self.indicators.output_gap = float(motion.output_gap)
        self.indicators.unemployment_rate = float(motion.unemployment)
        real_rate = compute_real_interest_rate(
            self.interest_rate, self.indicators.inflation_rate
        )
        self.reputation = update_reputation(
            self.reputation,
            previous_inflation,
            self.indicators.inflation_rate,
            self.indicators.unemployment_rate,
            real_rate,
        )

    def _record_initial_state(self):
        self.history.append(
            **self._history_values(
                quarter=0, events=(), aggregate_demand=None, aggregate_supply=None
            )
        )
        self._update_variables()

    def _record_quarter(self, motion, shocks, event_name):
        self.history.append(
            **self._history_values(
                quarter=self.current_quarter,
                events=(event_name,) if event_name else (),
                aggregate_demand=motion.aggregate_demand,
                aggregate_supply=motion.aggregate_supply,
                inflation_shock=shocks[0],
                demand_shock=shocks[1],
                natural_unemployment_shock=shocks[2],
                equilibrium_rate_shock=shocks[3],
            )
        )
        self._update_variables()

    def _history_values(self, quarter, events, aggregate_demand, aggregate_supply, **shocks):
        return {
            "quarter": quarter,
            "inflation_rate": self.indicators.inflation_rate,
            "gdp_growth": self.indicators.gdp_growth,
            "potential_growth": self.indicators.potential_growth,
            "output_gap": self.indicators.output_gap,
            "unemployment_rate": self.indicators.unemployment_rate,
            "natural_unemployment_rate": self.indicators.natural_unemployment_rate,
            "interest_rate": self.interest_rate,
            "real_interest_rate": compute_real_interest_rate(
                self.interest_rate, self.indicators.inflation_rate
            ),
            "equilibrium_real_rate": self.indicators.real_rate_eq,
            "reputation": self.reputation,
            "events": events,
            "aggregate_demand": aggregate_demand,
            "aggregate_supply": aggregate_supply,
            **shocks,
        }

    def _update_variables(self):
        """Update the narrow compatibility view read by existing GUI charts."""
        values = {
            "inflation_rate": self.indicators.inflation_rate,
            "unemployment_rate": self.indicators.unemployment_rate,
            "natural_unemployment_rate": self.indicators.natural_unemployment_rate,
            "interest_rate": self.interest_rate,
            "real_interest_rate": compute_real_interest_rate(
                self.interest_rate, self.indicators.inflation_rate
            ),
            "unemployment_gap": self.indicators.unemployment_rate
            - self.indicators.natural_unemployment_rate,
            "cb_reputation": self.reputation,
            "gdp_growth": self.indicators.gdp_growth,
            "potential_growth": self.indicators.potential_growth,
        }
        for name, value in values.items():
            self.variables.update(name, value)

    def apply_event_effects(self, effects):
        """Apply normalized event effects; retained as a public scenario API."""
        effects = self.event_engine.aggregate_effects(effects)
        self.indicators.inflation_rate += effects.get("inflation", 0.0)
        self.interest_rate += effects.get("interest_rate", 0.0)
        self.indicators.real_rate_eq += effects.get("real_rate_eq", 0.0)
        self.indicators.unemployment_rate += effects.get("unemployment", 0.0)
        if self.parameters.okun_coefficient > 0:
            self.indicators.output_gap -= (
                effects.get("unemployment", 0.0)
                / self.parameters.okun_coefficient
            )
        self.indicators.natural_unemployment_rate += effects.get(
            "natural_unemployment", 0.0
        )

    # Compatibility methods/properties used by the current desktop and web UIs.
    def enqueue_event(self, event):
        self.event_engine.enqueue(event)

    @property
    def events(self):
        return self.event_engine.events

    @property
    def effect_queue(self):
        return self.event_engine.effect_queue

    @property
    def past_events(self):
        return self.event_engine.past_events

    @past_events.setter
    def past_events(self, value):
        self.event_engine.past_events = value

    @property
    def last_event_quarter(self):
        return self.event_engine.last_event_quarter

    @last_event_quarter.setter
    def last_event_quarter(self, value):
        self.event_engine.last_event_quarter = value

    def _draw_cb_persona(self):
        return draw_persona()

    def adjust_interest_rate(self, new_rate):
        self.interest_rate = float(new_rate)

    def adjust_interest_rate_with_taylor(self):
        self.interest_rate = automated_rate(
            self.cb_persona, self.interest_rate, self.indicators
        )
        return self.interest_rate

    def get_state(self):
        return self.variables.values
