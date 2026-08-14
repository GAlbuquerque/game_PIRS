"""Editable model parameters for the economic simulation.

Keeping every coefficient in one dataclass makes the model inspectable today and
provides a simple source for point-and-click parameter editors in the future.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EconomyParameters:
    """All tunable values used by the economy's laws of motion."""

    # pi^e: constant expected-inflation intercept (beta0 in the AS curve).
    expected_inflation: float = 2.0
    # beta_pi: inflation response to the output gap in the Phillips curve.
    phillips_output_gap: float = 0.25
    # beta0: autonomous demand; later this can be estimated from history.
    demand_intercept: float = 2.0
    # beta_y: output response to the real-interest-rate gap (normally negative).
    demand_real_rate: float = -0.5
    # y^p: potential GDP growth used to define the output gap.
    potential_growth: float = 2.0
    # beta_u: Okun coefficient; positive output gaps reduce unemployment.
    okun_coefficient: float = 0.4
    # Long-run unemployment rate and slow quarterly reversion toward it.
    natural_unemployment_anchor: float = 5.0
    natural_unemployment_reversion: float = 0.02
    # Hard bounds prevent implausible numerical values from breaking the UI.
    minimum_unemployment: float = 1.0
    maximum_unemployment: float = 99.0
    minimum_natural_unemployment: float = 2.0
    # Quarterly standard deviations for inflation, demand, natural-rate, and r* shocks.
    shock_std_devs: tuple = (0.3, 0.2, 0.05, 0.1)
    # Contemporaneous correlations in the same order as shock_std_devs.
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
