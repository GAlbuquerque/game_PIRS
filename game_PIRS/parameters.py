"""Editable model parameters for the economic simulation.

Keeping every coefficient in one dataclass makes the model inspectable today and
provides a simple source for point-and-click parameter editors in the future.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EconomyParameters:
    """All tunable values used by the economy's laws of motion."""

    # Fallback pi^e used when the solver is called without inflation history.
    expected_inflation: float = 2.0
    # pi target used in beta_0,pi = alpha*pi_target + (1-alpha)*pi_(t-1).
    inflation_target: float = 2.0
    # beta_pi: inflation response to the output gap in the Phillips curve.
    phillips_output_gap: float = 0.05
    # Fallback shift to nominal-demand growth when rate history is unavailable.
    demand_intercept: float = 0.0
    # Effect of the current realized real-interest-rate gap on nominal demand.
    demand_real_rate: float = -0.6
    # Weights on trailing real-rate gaps in nominal-demand growth.
    demand_intercept_weight_10: float = -3.0
    demand_intercept_weight_20: float = -1.2
    # Quarterly turns report annualized growth, so gap changes are divided by 4.
    periods_per_year: int = 4
    # y^p: potential GDP growth used to define the output gap.
    potential_growth: float = 2.0
    # beta_u: Okun coefficient; positive output gaps reduce unemployment.
    okun_coefficient: float = 0.4
    # Long-run unemployment rate and slow quarterly reversion toward it.
    natural_unemployment_anchor: float = 5.0
    natural_unemployment_reversion: float = 0.02
    # Hard bounds prevent implausible numerical values from breaking the UI.
    minimum_inflation: float = -99.0
    minimum_unemployment: float = 1.0
    # At this unemployment rate AS becomes vertical (an output ceiling).
    vertical_supply_unemployment: float = 2.0
    maximum_unemployment: float = 99.0
    minimum_natural_unemployment: float = 2.0
    # Numerical solver accuracy: AD and AS errors must both be below this value.
    solver_tolerance: float = 1e-9
    # Maximum Newton iterations before reporting that the curves did not converge.
    solver_max_iterations: int = 50
    # Small change used to estimate how equation errors respond to pi and y.
    solver_step_size: float = 1e-5
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
