"""Generation and storage of complete quarterly economic history."""

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class HistoryEntry:
    quarter: int
    inflation_rate: float
    gdp_growth: float
    potential_growth: float
    output_gap: float
    unemployment_rate: float
    natural_unemployment_rate: float
    interest_rate: float
    real_interest_rate: float
    equilibrium_real_rate: float
    reputation: float
    events: tuple = ()


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

    @classmethod
    def generate_random(cls, quarters, initial, parameters, rng=None):
        """Generate a plausible pre-game history using the same AD-AS laws."""
        import numpy as np
        from laws_of_motion import solve_ad_as
        from utils import compute_real_interest_rate

        rng = rng or np.random.default_rng()
        history = cls()
        unemployment = initial.unemployment_rate
        interest = initial.target_inflation_rate + initial.real_rate_eq
        for quarter in range(-quarters, 0):
            result = solve_ad_as(
                interest, initial.real_rate_eq, unemployment,
                rng.normal(0, parameters.std_devs[0]),
                rng.normal(0, parameters.std_devs[1]), parameters,
            )
            unemployment = result.unemployment
            history.append(
                quarter=quarter, inflation_rate=result.inflation,
                gdp_growth=result.output_growth, potential_growth=parameters.potential_growth,
                output_gap=result.output_gap, unemployment_rate=unemployment,
                natural_unemployment_rate=initial.natural_unemployment_rate,
                interest_rate=interest,
                real_interest_rate=compute_real_interest_rate(interest, result.inflation),
                equilibrium_real_rate=initial.real_rate_eq, reputation=0.8, events=(),
            )
        return history
