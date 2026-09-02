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
    # k maps central-bank reputation into the weight placed on its target.
    reputation_expectation_coefficient: float = 0.1
    # An optional labor-market objective; None represents a pure inflation target.
    unemployment_target: float | None = None
    # beta: weight on expected inflation in the quarterly Phillips curve.
    inflation_expectation_discount: float = 1.0
    # kappa: inflation response to a positive output gap.
    phillips_output_gap: float = 0.1
    # A negative gap has half the ordinary Phillips-curve slope.
    negative_gap_slope_ratio: float = 0.5
    # Negative raw inflation is multiplied by this adjustment ratio.
    deflation_supply_slope_ratio: float = 0.5
    # phi: expected persistence of the most recently observed output gap.
    output_gap_expectation_persistence: float = 0.8
    # sigma in the dynamic IS equation (inverse intertemporal elasticity).
    intertemporal_elasticity_inverse: float = 1.0
    # rho: persistence of the effective, lagged real-rate gap.
    interest_rate_pressure_persistence: float = 0.5
    # Scale on the contractionary IS effect of a positive effective rate gap.
    demand_interest_rate_pressure: float = 1.0
    # Quarterly gap changes are multiplied by 4 to report annualized growth.
    periods_per_year: int = 4
    # y^p: potential GDP growth used to define the output gap.
    potential_growth: float = 2.0
    # beta_u: Okun coefficient; positive output gaps reduce unemployment.
    okun_coefficient: float = 0.7
    # Long-run unemployment rate and slow quarterly reversion toward it.
    natural_unemployment_anchor: float = 5.0
    natural_unemployment_reversion: float = 0.02
    # Long-run equilibrium real rate and its slow quarterly reversion speed.
    equilibrium_real_rate_anchor: float = 0.5
    equilibrium_real_rate_reversion: float = 0.02
    # Hard bounds prevent implausible numerical values from breaking the UI.
    minimum_inflation: float = -99.0
    minimum_unemployment: float = 1.0
    # Legacy field accepted when loading older direct EconomyParameters calls.
    vertical_supply_unemployment: float = 1.0
    maximum_unemployment: float = 99.0
    minimum_natural_unemployment: float = 2.0
    # Legacy numerical-solver settings retained for old settings codes.
    solver_tolerance: float = 1e-9
    # Maximum Newton iterations before reporting that the curves did not converge.
    solver_max_iterations: int = 50
    # Small change used to estimate how equation errors respond to pi and y.
    solver_step_size: float = 1e-5
    # Quarterly standard deviations for inflation, demand, natural-rate, and r* shocks.
    shock_std_devs: tuple = (0.3, 0.2, 0.05, 0.1)
    # Multiplier applied to every event probability (0 disables random events).
    event_probability_scale: float = 1.0
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
