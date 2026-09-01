"""Editable parameters for the quarterly New Keynesian simulation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EconomyParameters:
    """All tunable values, named to match the equations in the model guide."""

    expected_inflation: float = 2.0
    inflation_target: float = 2.0
    reputation_expectation_coefficient: float = 0.1
    unemployment_target: float | None = None

    # beta and kappa in pi_t = beta E_t[pi_(t+1)] + kappa x_t + shock.
    inflation_expectation_discount: float = 1.0
    phillips_output_gap: float = 0.1
    negative_gap_slope_ratio: float = 0.5
    deflation_adjustment_ratio: float = 0.5

    # phi, rho, and sigma in the expected-gap, effective-rate, and IS equations.
    output_gap_expectation: float = 0.8
    real_rate_persistence: float = 0.5
    intertemporal_elasticity_inverse: float = 1.0

    periods_per_year: int = 4
    potential_growth: float = 2.0
    okun_coefficient: float = 0.7
    natural_unemployment_anchor: float = 5.0
    natural_unemployment_reversion: float = 0.02
    equilibrium_real_rate_anchor: float = 0.5
    equilibrium_real_rate_reversion: float = 0.02

    minimum_inflation: float = -99.0
    minimum_unemployment: float = 1.0
    maximum_unemployment: float = 99.0
    minimum_natural_unemployment: float = 2.0

    shock_std_devs: tuple = (0.3, 0.2, 0.05, 0.1)
    event_probability_scale: float = 1.0
    shock_correlations: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 0.0, 0.1, 0.0],
                [0.0, 1.0, 0.2, 0.0],
                [0.1, 0.2, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )

    @property
    def std_devs(self):
        return np.asarray(self.shock_std_devs, dtype=float)

    # Read-only aliases allow older integrations to inspect renamed settings.
    @property
    def interest_rate_pressure_persistence(self):
        return self.real_rate_persistence

    @property
    def demand_interest_rate_pressure(self):
        return self.intertemporal_elasticity_inverse

    @property
    def deflation_supply_slope_ratio(self):
        return self.deflation_adjustment_ratio

    @property
    def vertical_supply_unemployment(self):
        return self.minimum_unemployment
