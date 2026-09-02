#!/usr/bin/env python3
"""High-level coordinator for one quarter of the economic simulation."""

from dataclasses import replace

import numpy as np

from event_engine import EventEngine
from history import EconomicHistory
from indicators import EconomicIndicators
from laws_of_motion import (
    calculate_expected_inflation,
    calculate_interest_rate_pressure,
    calculate_real_interest_rate,
    calculate_quarter_outcome,
)
from parameters import EconomyParameters
from personas import automated_rate, draw_persona
from reputation import update_reputation
from shocks import generate_shocks
from variables import Variables


class Economy:
    """Coordinate events, shocks, laws of motion, reputation, and history."""

    EVENT_HORIZON = 8
    PLAYER_EVENT_SCHEDULES = {
        "quantitative_easing": {
            "demand": (0.5, 1.0, 0.65, 0.4, 0.25, 0.15, 0.1, 0.05),
            "inflation": (0.05, 0.1, 0.065, 0.04, 0.025, 0.015, 0.01, 0.005),
        },
        "high_rate_guidance": {"rate_pressure": (1.0,) * 4},
        "low_rate_guidance": {"rate_pressure": (-1.0,) * 4},
    }

    def __init__(
        self,
        initial_state=None,
        difficulty="central_banker",
        scenario=None,
        parameters=None,
        random_history_quarters=0,
        minimum_interest_rate=0.0,
    ):
        self.parameters = parameters or EconomyParameters()
        self.minimum_interest_rate = float(minimum_interest_rate)
        self.difficulty = difficulty
        self.shock_sd_scale = self._difficulty_shock_scale(difficulty)
        self.indicators = initial_state or EconomicIndicators.generate_random_initial_state()
        if scenario is not None:
            self.indicators = replace(self.indicators, **scenario)
        self.indicators.target_inflation_rate = self.parameters.inflation_target
        if self.indicators.output_gap is None:
            self.indicators.output_gap = (
                self.indicators.natural_unemployment_rate
                - self.indicators.unemployment_rate
            ) / self.parameters.okun_coefficient

        self.interest_rate = max(
            float(np.random.normal(0.5, 2)), self.minimum_interest_rate
        )
        self.reputation = 0.8
        self.expected_inflation = calculate_expected_inflation(
            self.indicators.inflation_rate,
            self.indicators.target_inflation_rate,
            self.reputation,
            self.parameters,
        )
        self.cb_persona = draw_persona()
        self.current_quarter = 1
        self.max_quarters = 50
        self.offset = 0
        self.player_start_turn = 40
        self.interest_rate_pressure = 0.0
        self.player_event_queue = []
        self.player_event_last_used = {}
        self.player_event_used_quarter = None

        self.event_engine = EventEngine(
            difficulty=difficulty,
            horizon=self.EVENT_HORIZON,
            cooldown_quarters=self._difficulty_event_cooldown(difficulty),
            probability_scale=self.parameters.event_probability_scale,
        )
        self.history = EconomicHistory.generate_random(
            random_history_quarters, self.indicators, self.parameters
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

        player_effects = self._current_player_event_effects()
        shocks = generate_shocks(
            self.parameters.shock_correlations,
            self.parameters.std_devs * self.shock_sd_scale,
        )
        previous_inflation = self.indicators.inflation_rate
        self._apply_background_shocks(shocks)
        real_rates = self.history.series("real_interest_rate")
        equilibrium_real_rates = self.history.series("equilibrium_real_rate")
        self.interest_rate_pressure = calculate_interest_rate_pressure(
            real_rates,
            equilibrium_real_rates,
            self.interest_rate_pressure,
            self.parameters,
        )
        effective_rate_pressure = (
            self.interest_rate_pressure + player_effects.get("rate_pressure", 0.0)
        )
        motion = calculate_quarter_outcome(
            natural_unemployment=self.indicators.natural_unemployment_rate,
            inflation_shock=shocks[0] + event_inflation + player_effects.get("inflation", 0.0),
            demand_shock=shocks[1] + player_effects.get("demand", 0.0),
            parameters=self.parameters,
            previous_inflation=previous_inflation,
            target_inflation=self.indicators.target_inflation_rate,
            reputation=self.reputation,
            previous_output_gap=self.indicators.output_gap,
            interest_rate_pressure=effective_rate_pressure,
        )
        self._commit_motion(motion, previous_inflation)
        recorded_shocks = shocks.copy()
        recorded_shocks[0] += event_inflation + player_effects.get("inflation", 0.0)
        recorded_shocks[1] += player_effects.get("demand", 0.0)
        self._record_quarter(motion, recorded_shocks, outcome.name)
        self.current_quarter += 1
        return {
            "event": outcome.description,
            "event_name": outcome.name,
            "gap_effect": motion.output_gap,  # Legacy result key used by the UI.
            "shocks": shocks.tolist(),
        }

    def trigger_player_event(self, event_name):
        """Schedule a discretionary player action, if it is currently available."""
        if self.difficulty != "central_banker":
            return False, "Player-triggered events are only available in Central Banker mode."
        if event_name not in self.PLAYER_EVENT_SCHEDULES:
            return False, "Unknown player event."
        if self.player_event_used_quarter == self.current_quarter:
            return False, "Only one player event may be used per quarter."
        if event_name == "high_rate_guidance" and self.reputation <= 0.7:
            return False, "High-rate guidance requires reputation above 0.70."
        if event_name in ("high_rate_guidance", "low_rate_guidance"):
            last_used = self.player_event_last_used.get(event_name)
            if last_used is not None and self.current_quarter - last_used < 4:
                return False, "This announcement has a four-quarter cooldown."

        self.player_event_queue.append({
            "name": event_name,
            "start_quarter": self.current_quarter,
        })
        self.player_event_last_used[event_name] = self.current_quarter
        self.player_event_used_quarter = self.current_quarter
        return True, "Player event scheduled."

    def player_event_status(self, event_name):
        """Return whether an action can be selected and a UI-ready reason."""
        if self.difficulty != "central_banker":
            return False, "Central Banker difficulty only"
        if self.player_event_used_quarter == self.current_quarter:
            return False, "An action was already used this quarter"
        if event_name == "high_rate_guidance" and self.reputation <= 0.7:
            return False, "Requires reputation above 0.70"
        if event_name in ("high_rate_guidance", "low_rate_guidance"):
            last_used = self.player_event_last_used.get(event_name)
            if last_used is not None:
                remaining = 4 - (self.current_quarter - last_used)
                if remaining > 0:
                    return False, f"Cooldown: {remaining} quarter(s) remaining"
        return True, "Available"

    def _current_player_event_effects(self):
        effects = {"demand": 0.0, "inflation": 0.0, "rate_pressure": 0.0}
        active = []
        for queued in self.player_event_queue:
            age = self.current_quarter - queued["start_quarter"]
            schedule = self.PLAYER_EVENT_SCHEDULES[queued["name"]]
            still_active = False
            for effect_name, values in schedule.items():
                if 0 <= age < len(values):
                    effects[effect_name] += values[age]
                    still_active = True
            if still_active:
                active.append(queued)
        self.player_event_queue = active
        return effects

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
        """Evolve natural unemployment and r* before calculating this quarter."""
        p = self.parameters
        natural_drift = -p.natural_unemployment_reversion * (
            self.indicators.natural_unemployment_rate
            - p.natural_unemployment_anchor
        )
        self.indicators.natural_unemployment_rate = max(
            p.minimum_natural_unemployment,
            self.indicators.natural_unemployment_rate + natural_drift + shocks[2],
        )
        equilibrium_rate_drift = -p.equilibrium_real_rate_reversion * (
            self.indicators.real_rate_eq - p.equilibrium_real_rate_anchor
        )
        self.indicators.real_rate_eq += equilibrium_rate_drift + shocks[3]

    def _commit_motion(self, motion, previous_inflation):
        self.expected_inflation = float(motion.expected_inflation)
        self.indicators.inflation_rate = max(
            float(motion.inflation), self.parameters.minimum_inflation
        )
        self.indicators.output_gap = float(motion.output_gap)
        self.indicators.unemployment_rate = float(motion.unemployment)
        real_rate = calculate_real_interest_rate(
            self.interest_rate, self.expected_inflation
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
                quarter=0, events=()
            )
        )
        self._update_variables()

    def _record_quarter(self, motion, shocks, event_name):
        self.history.append(
            **self._history_values(
                quarter=self.current_quarter,
                events=(event_name,) if event_name else (),
                inflation_shock=shocks[0],
                demand_shock=shocks[1],
                natural_unemployment_shock=shocks[2],
                equilibrium_rate_shock=shocks[3],
            )
        )
        self._update_variables()

    def _history_values(self, quarter, events, **shocks):
        return {
            "quarter": quarter,
            "inflation_rate": self.indicators.inflation_rate,
            "output_gap": self.indicators.output_gap,
            "unemployment_rate": self.indicators.unemployment_rate,
            "natural_unemployment_rate": self.indicators.natural_unemployment_rate,
            "interest_rate": self.interest_rate,
            "expected_inflation": self.expected_inflation,
            "real_interest_rate": calculate_real_interest_rate(
                self.interest_rate, self.expected_inflation
            ),
            "equilibrium_real_rate": self.indicators.real_rate_eq,
            "interest_rate_pressure": self.interest_rate_pressure,
            "reputation": self.reputation,
            "events": events,
            **shocks,
        }

    def _update_variables(self):
        """Update the narrow compatibility view read by existing GUI charts."""
        values = {
            "inflation_rate": self.indicators.inflation_rate,
            "unemployment_rate": self.indicators.unemployment_rate,
            "natural_unemployment_rate": self.indicators.natural_unemployment_rate,
            "interest_rate": self.interest_rate,
            "real_interest_rate": calculate_real_interest_rate(
                self.interest_rate, self.expected_inflation
            ),
            "unemployment_gap": self.indicators.unemployment_rate
            - self.indicators.natural_unemployment_rate,
            "cb_reputation": self.reputation,
        }
        for name, value in values.items():
            self.variables.update(name, value)

    def apply_event_effects(self, effects):
        """Apply normalized event effects; retained as a public scenario API."""
        effects = self.event_engine.aggregate_effects(effects)
        self.indicators.inflation_rate += effects.get("inflation", 0.0)
        self.interest_rate += effects.get("interest_rate", 0.0)
        self.interest_rate = max(self.interest_rate, self.minimum_interest_rate)
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
        self.interest_rate = max(float(new_rate), self.minimum_interest_rate)

    def adjust_interest_rate_with_taylor(self):
        self.interest_rate = automated_rate(
            self.cb_persona, self.interest_rate, self.indicators
        )
        self.interest_rate = max(self.interest_rate, self.minimum_interest_rate)
        return self.interest_rate

    def get_state(self):
        return self.variables.values
