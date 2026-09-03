"""Readable, equation-by-equation implementation of the quarterly model.

The order of the functions follows ``docs/economic_model.tex``: expectations,
the lagged real-rate gap, the dynamic IS equation, capacity, the Phillips curve,
and Okun's law.  Keeping those steps separate makes the simulation easier to
compare with the model and lets each equation be tested on its own.
"""

from dataclasses import dataclass

import numpy as np

from parameters import EconomyParameters


@dataclass(frozen=True)
class ModelResult:
    """The results of applying this quarter's equations in model order."""

    inflation: float
    unemployment: float
    output_gap: float
    expected_inflation: float = 2.0


def calculate_expected_inflation(
    previous_inflation, target_inflation, reputation, parameters
):
    """Return the one-quarter-ahead inflation expectation.

    Reputation determines how much weight agents put on the announced target;
    the remaining weight goes on the latest inflation observation.
    """
    target = (
        parameters.inflation_target
        if target_inflation is None
        else target_inflation
    )
    if previous_inflation is None:
        return float(target)
    anchoring_strength = parameters.reputation_expectation_coefficient
    if not 0.0 <= anchoring_strength <= 1.0:
        raise ValueError("reputation expectation coefficient must be between 0 and 1")
    if reputation is not None and not 0.0 <= reputation <= 1.0:
        raise ValueError("reputation must be between 0 and 1")
    target_weight = (
        0.0
        if reputation is None
        else reputation * anchoring_strength
    )
    return target_weight * target + (1.0 - target_weight) * previous_inflation


def calculate_expected_output_gap(previous_output_gap, parameters):
    """Apply the game's simple expectation that an existing gap will shrink."""
    persistence = parameters.output_gap_expectation_persistence
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("output-gap expectation persistence must be between 0 and 1")
    return persistence * previous_output_gap


def calculate_real_interest_rate(nominal_interest_rate, expected_inflation):
    """Calculate the ex-ante real rate chosen in a quarter: r_t = i_t-E_t pi_(t+1)."""
    return float(nominal_interest_rate - expected_inflation)


def calculate_real_interest_rate_gap(real_interest_rate, equilibrium_real_rate):
    """Measure monetary restraint relative to the natural real rate."""
    return float(real_interest_rate - equilibrium_real_rate)


def calculate_interest_rate_pressure(
    real_interest_rates, equilibrium_real_rates, previous_pressure, parameters
):
    """Update the effective real-rate gap with the rate observed in t-1.

    At the start of quarter t, history ends in t-1.  Thus the final stored rate
    gap is the new observation in

        R_t = rho R_(t-1) + (1-rho) (r_(t-1)-r*_(t-1)).

    The rate selected during t is deliberately absent; it is stored at the end
    of the turn and first affects output in t+1.
    """
    rates = np.asarray(real_interest_rates, dtype=float)
    natural_rates = np.asarray(equilibrium_real_rates, dtype=float)
    if rates.shape != natural_rates.shape:
        raise ValueError(
            "real and equilibrium real interest rate histories must have equal length"
        )

    rho = parameters.interest_rate_pressure_persistence
    if not 0.0 <= rho <= 1.0:
        raise ValueError("interest-rate pressure persistence must be between 0 and 1")
    if rates.size == 0:
        return float(previous_pressure)

    lagged_rate_gap = calculate_real_interest_rate_gap(rates[-1], natural_rates[-1])
    return float(rho * previous_pressure + (1.0 - rho) * lagged_rate_gap)


def dynamic_is_equation(
    expected_future_output_gap, effective_real_rate_gap, demand_shock, parameters
):
    """Return unconstrained output from the modified dynamic IS equation."""
    if parameters.intertemporal_elasticity_inverse <= 0:
        raise ValueError("intertemporal_elasticity_inverse must be positive")
    interest_effect = (
        effective_real_rate_gap / parameters.intertemporal_elasticity_inverse
    )
    return float(expected_future_output_gap - interest_effect + demand_shock)


def calculate_maximum_output_gap(natural_unemployment, parameters):
    """Infer the largest feasible output gap from the 1 percent jobless floor."""
    if parameters.okun_coefficient <= 0:
        raise ValueError("okun_coefficient must be positive")
    return float(
        (natural_unemployment - parameters.minimum_unemployment)
        / parameters.okun_coefficient
    )


def apply_output_capacity(unconstrained_output_gap, maximum_output_gap):
    """Clip only the upper side of IS at the capacity implied by Okun's law."""
    return float(min(unconstrained_output_gap, maximum_output_gap))


def phillips_curve_gap_effect(output_gap, parameters):
    """Return the Phillips-curve contribution of the output gap."""
    slope = parameters.phillips_output_gap
    if slope < 0:
        raise ValueError("phillips_output_gap cannot be negative")
    return float(slope * output_gap)


def new_keynesian_phillips_curve(
    expected_inflation, output_gap, inflation_shock, parameters
):
    """Calculate inflation before the deflation guardrail is applied."""
    discount = parameters.inflation_expectation_discount
    if not 0.0 < discount <= 1.0:
        raise ValueError("inflation expectation discount must be above 0 and at most 1")
    expected_component = discount * expected_inflation
    return float(
        expected_component
        + phillips_curve_gap_effect(output_gap, parameters)
        + inflation_shock
    )


def apply_inflation_floor(inflation, parameters):
    """Impose the inflation floor without treating deflation differently."""
    return float(max(parameters.minimum_inflation, inflation))


def okuns_law(natural_unemployment, output_gap, parameters):
    """Translate the output-level gap into unemployment after output is known."""
    return float(natural_unemployment - parameters.okun_coefficient * output_gap)


def calculate_quarter_outcome(
    natural_unemployment,
    inflation_shock,
    demand_shock,
    parameters: EconomyParameters,
    *,
    previous_inflation=None,
    target_inflation=None,
    reputation=None,
    previous_output_gap=0.0,
    interest_rate_pressure=0.0,
):
    """Apply the model equations sequentially for one quarter.

    Current policy is intentionally absent.  The coordinator records today's
    ex-ante real-rate gap for use in next quarter's effective rate gap.
    """
    expected_inflation = calculate_expected_inflation(
        previous_inflation, target_inflation, reputation, parameters
    )
    expected_future_gap = calculate_expected_output_gap(
        previous_output_gap, parameters
    )
    unconstrained_output_gap = dynamic_is_equation(
        expected_future_gap,
        interest_rate_pressure,
        demand_shock,
        parameters,
    )

    capacity = calculate_maximum_output_gap(natural_unemployment, parameters)
    output_gap = apply_output_capacity(unconstrained_output_gap, capacity)

    raw_inflation = new_keynesian_phillips_curve(
        expected_inflation, output_gap, inflation_shock, parameters
    )
    inflation = apply_inflation_floor(raw_inflation, parameters)
    unemployment = okuns_law(natural_unemployment, output_gap, parameters)

    return ModelResult(
        inflation=inflation,
        unemployment=unemployment,
        output_gap=output_gap,
        expected_inflation=float(expected_inflation),
    )
