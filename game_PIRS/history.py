"""Generation and storage of complete quarterly economic history."""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class HistoryEntry:
    quarter: int
    inflation_rate: float
    output_gap: float
    unemployment_rate: float
    natural_unemployment_rate: float
    interest_rate: float
    real_interest_rate: float
    equilibrium_real_rate: float
    interest_rate_pressure: float
    reputation: float
    expected_inflation: float = 2.0
    events: tuple = ()
    inflation_shock: float = 0.0
    demand_shock: float = 0.0
    natural_unemployment_shock: float = 0.0
    equilibrium_rate_shock: float = 0.0


class EconomicHistory:
    """Append-only vector of every relevant variable and event."""

    def __init__(self, entries=None):
        self.entries = list(entries or [])

    def append(self, **values):
        values["events"] = tuple(values.get("events", ()))
        self.entries.append(HistoryEntry(**values))

    def series(self, name):
        return [getattr(entry, name) for entry in self.entries]

    def to_frame(self):
        return pd.DataFrame(asdict(entry) for entry in self.entries)

    def event_snapshot(self, quarter_user, past_events):
        """Return the legacy key names consumed by event probability functions."""
        return {
            "inflation_rate": self.series("inflation_rate"),
            "interest_rate": self.series("interest_rate"),
            "real_rate_eq": self.series("real_interest_rate"),
            "unemployment_rate": self.series("unemployment_rate"),
            "natural_unemployment_rate": self.series("natural_unemployment_rate"),
            "reputation_history": self.series("reputation"),
            "past_events": list(past_events),
            "quarter_user": quarter_user,
        }

    @classmethod
    def generate_random(cls, quarters, initial, parameters, rng=None):
        """Generate a plausible pre-game history using the same model equations."""
        import numpy as np
        from laws_of_motion import (
            calculate_interest_rate_pressure,
            calculate_real_interest_rate,
            calculate_quarter_outcome,
        )

        rng = rng or np.random.default_rng()
        history = cls()
        unemployment = initial.unemployment_rate
        output_gap = initial.output_gap
        if output_gap is None:
            output_gap = (
                initial.natural_unemployment_rate - unemployment
            ) / parameters.okun_coefficient
        inflation = initial.inflation_rate
        interest = initial.target_inflation_rate + initial.real_rate_eq
        interest_rate_pressure = 0.0
        for quarter in range(-quarters, 0):
            interest_rate_pressure = calculate_interest_rate_pressure(
                history.series("real_interest_rate"),
                history.series("equilibrium_real_rate"),
                interest_rate_pressure,
                parameters,
            )
            result = calculate_quarter_outcome(
                initial.natural_unemployment_rate,
                rng.normal(0, parameters.std_devs[0]),
                rng.normal(0, parameters.std_devs[1]), parameters,
                previous_inflation=inflation,
                target_inflation=initial.target_inflation_rate,
                reputation=0.8,
                previous_output_gap=output_gap,
                interest_rate_pressure=interest_rate_pressure,
            )
            unemployment = result.unemployment
            output_gap = result.output_gap
            inflation = result.inflation
            history.append(
                quarter=quarter, inflation_rate=result.inflation,
                output_gap=result.output_gap, unemployment_rate=unemployment,
                natural_unemployment_rate=initial.natural_unemployment_rate,
                interest_rate=interest,
                expected_inflation=result.expected_inflation,
                real_interest_rate=calculate_real_interest_rate(
                    interest, result.expected_inflation
                ),
                equilibrium_real_rate=initial.real_rate_eq, reputation=0.8, events=(),
                interest_rate_pressure=interest_rate_pressure,
            )
        return history
