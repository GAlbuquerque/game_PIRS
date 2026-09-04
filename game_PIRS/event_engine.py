"""Event selection, cooldowns, schedules, and effect aggregation."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from events import initialize_events


@dataclass(frozen=True)
class EventOutcome:
    """The event selected this quarter and all effects due this quarter."""

    description: str | None
    name: str | None
    effects: dict


class EventEngine:
    """Own all mutable event state so the economy only coordinates a turn."""

    FISCAL_EVENTS = {"Fiscal Deficit", "Spending Wave", "Fiscal Surplus"}

    def __init__(
        self,
        difficulty,
        horizon=8,
        cooldown_quarters=0,
        events=None,
        probability_scale=1.0,
        okun_coefficient=0.7,
    ):
        self.difficulty = difficulty
        self.horizon = horizon
        self.cooldown_quarters = cooldown_quarters
        self.events = list(
            events
            if events is not None
            else initialize_events(okun_coefficient=okun_coefficient)
        )
        self.probability_scale = max(0.0, float(probability_scale))
        self.effect_queue = [defaultdict(float) for _ in range(horizon)]
        self.past_events = []
        self.last_event_quarter = -10_000

    def advance(self, history, current_quarter, player_start_turn=40):
        """Select an event, schedule it, and consume the current effect slice."""
        event = self.select_event(history, current_quarter, player_start_turn)
        names = []
        if event is not None:
            names.append(event.name)
            self.enqueue(event)
        self.past_events.append(names)
        self.past_events = self.past_events[-self.horizon :]
        effects = self.consume_effects()
        return EventOutcome(
            description=event.description if event else None,
            name=event.name if event else None,
            effects=effects,
        )

    def select_event(self, history, current_quarter, player_start_turn=40):
        """Select at most one eligible event using its configured probability."""
        cooperative = self._select_cooperative_fiscal_event(
            history, current_quarter, player_start_turn
        )
        if cooperative is not None:
            return cooperative
        if current_quarter - self.last_event_quarter < self.cooldown_quarters:
            return None

        fired = []
        for event in self.events:
            allowed = getattr(event, "allowed_difficulties", None)
            if allowed is not None and self.difficulty not in allowed:
                continue
            probability = float(event.get_probability(history)) * self.probability_scale
            if np.random.rand() < max(0.0, min(1.0, probability)):
                fired.append(event)
        if not fired:
            return None
        self.last_event_quarter = current_quarter
        return np.random.choice(fired)

    def _select_cooperative_fiscal_event(self, history, current_quarter, player_start_turn):
        if self.difficulty != "principles" or history.get("quarter_user", 0) <= player_start_turn:
            return None
        if self._recent_fiscal_event_count(4):
            return None
        inflation = history.get("inflation_rate", [0.0])[-1]
        unemployment = history.get("unemployment_rate", [0.0])[-1]
        natural = history.get("natural_unemployment_rate", [0.0])[-1]
        target = None
        if inflation < 0 and unemployment > natural:
            target = "Fiscal Deficit" if np.random.rand() < 0.5 else "Spending Wave"
        elif inflation > 10 and np.random.rand() < 0.5:
            target = "Fiscal Surplus"
        if target is None:
            return None
        self.last_event_quarter = current_quarter
        return next((event for event in self.events if event.name == target), None)

    def _recent_fiscal_event_count(self, within):
        return sum(
            name in self.FISCAL_EVENTS
            for quarter in self.past_events[-within:]
            for name in (quarter if isinstance(quarter, list) else [quarter])
        )

    def enqueue(self, event):
        """Add an event's multi-quarter schedule to the aggregate queue."""
        for indicator, sequence in event.effects_schedule.items():
            for quarter, value in enumerate(sequence[: self.horizon]):
                self.effect_queue[quarter][indicator] += float(value or 0.0)

    def consume_effects(self):
        effects = dict(self.effect_queue.pop(0))
        self.effect_queue.append(defaultdict(float))
        return effects

    @staticmethod
    def aggregate_effects(effects):
        """Normalize either one effect dictionary or a list of dictionaries."""
        aggregate = defaultdict(float)
        effect_sets = effects if isinstance(effects, list) else [effects]
        for effect_set in effect_sets:
            for key, value in (effect_set or {}).items():
                aggregate[key] += float(value or 0.0)
        return dict(aggregate)
