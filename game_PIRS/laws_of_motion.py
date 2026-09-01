"""Quarterly laws of motion for the modified New Keynesian model.

Each model equation has its own small function.  Keeping the equations
separate makes the order of the simulation visible to readers who know the
economics but do not need to know numerical-solution techniques.
"""

from dataclasses import dataclass

from parameters import EconomyParameters


@dataclass(frozen=True)
class MotionResult:
    """Values produced by one pass through the model's equations."""

    inflation: float
    output_growth: float
    unemployment: float
    output_gap: float
    aggregate_demand: float
    aggregate_supply: float
    expected_inflation: float = 2.0


def calculate_expected_inflation(
    previous_inflation, target_inflation, reputation, parameters
):
    """Return E_t[pi_(t+1)], combining the target and observed inflation."""
    if previous_inflation is None:
        return parameters.expected_inflation
    target = parameters.inflation_target if target_inflation is None else target_inflation
    target_weight = 0.0 if reputation is None else reputation * parameters.reputation_expectation_coefficient
    target_weight = min(1.0, max(0.0, target_weight))
    return target_weight * target + (1.0 - target_weight) * previous_inflation


def calculate_real_interest_rate(player_interest_rate, expected_inflation):
    """Implement r_t = i_t - E_t[pi_(t+1)]."""
    return float(player_interest_rate - expected_inflation)


def calculate_effective_real_rate(previous_effective_rate, lagged_real_rate, parameters):
    """Implement R_t = rho R_(t-1) + (1-rho) r_(t-1)."""
    rho = parameters.real_rate_persistence
    if not 0.0 <= rho <= 1.0:
        raise ValueError("real_rate_persistence must be between 0 and 1")
    return float(rho * previous_effective_rate + (1.0 - rho) * lagged_real_rate)


def calculate_interest_rate_pressure(
    real_interest_rates, equilibrium_real_rates, previous_pressure, parameters
):
    """Compatibility wrapper for old callers of the effective-rate equation.

    The equilibrium-rate history is accepted because it was part of the old
    interface, but R_t smooths real rates themselves, not real-rate gaps.
    """
    if len(real_interest_rates) != len(equilibrium_real_rates):
        raise ValueError("real and equilibrium real interest rate histories must have equal length")
    if not real_interest_rates:
        return float(previous_pressure)
    return calculate_effective_real_rate(
        previous_pressure, real_interest_rates[-1], parameters
    )


def expected_output_gap(previous_output_gap, parameters):
    """Implement E_t[x_(t+1)] = phi x_(t-1)."""
    return float(parameters.output_gap_expectation * previous_output_gap)


def dynamic_is_equation(
    expected_future_output_gap,
    effective_real_rate,
    equilibrium_real_rate,
    demand_shock,
    parameters,
):
    """Implement the modified dynamic IS equation before its capacity guardrail."""
    if parameters.intertemporal_elasticity_inverse <= 0.0:
        raise ValueError("intertemporal_elasticity_inverse must be positive")
    real_rate_gap = effective_real_rate - equilibrium_real_rate
    return float(
        expected_future_output_gap
        - real_rate_gap / parameters.intertemporal_elasticity_inverse
        + demand_shock
    )


def maximum_output_gap(natural_unemployment, parameters):
    """Use Okun's law to find the gap consistent with the unemployment floor."""
    if parameters.okun_coefficient <= 0.0:
        raise ValueError("okun_coefficient must be positive")
    return float(
        (natural_unemployment - parameters.minimum_unemployment)
        / parameters.okun_coefficient
    )


def apply_output_capacity(output_gap, natural_unemployment, parameters):
    """Cap the IS result at the gap implied by one-percent unemployment."""
    return min(float(output_gap), maximum_output_gap(natural_unemployment, parameters))


def new_keynesian_phillips_curve(
    expected_inflation, output_gap, inflation_shock, parameters
):
    """Implement the NKPC, with a flatter slope when the output gap is negative."""
    gap_slope = parameters.phillips_output_gap
    if output_gap < 0.0:
        gap_slope *= parameters.negative_gap_slope_ratio
    return float(
        parameters.inflation_expectation_discount * expected_inflation
        + gap_slope * output_gap
        + inflation_shock
    )


def slow_deflation(inflation, parameters):
    """Reduce negative inflation in magnitude to represent downward rigidity."""
    if inflation < 0.0:
        return float(inflation * parameters.deflation_adjustment_ratio)
    return float(inflation)


def apply_inflation_floor(inflation, parameters):
    """Prevent the simulated price level from falling by 100 percent or more."""
    return max(float(inflation), parameters.minimum_inflation)


def okuns_law(natural_unemployment, output_gap, parameters):
    """Implement u_t = u_t^n - beta_u x_t after output is determined."""
    unemployment = natural_unemployment - parameters.okun_coefficient * output_gap
    return min(parameters.maximum_unemployment, max(parameters.minimum_unemployment, unemployment))


def calculate_output_growth(output_gap, previous_output_gap, parameters):
    """Convert the quarterly change in the level gap to annualized GDP growth."""
    return float(
        parameters.potential_growth
        + parameters.periods_per_year * (output_gap - previous_output_gap)
    )


def solve_ad_as(
    player_interest_rate,
    equilibrium_real_rate,
    previous_unemployment,
    inflation_shock,
    demand_shock,
    parameters: EconomyParameters,
    *,
    previous_inflation=None,
    target_inflation=None,
    reputation=None,
    natural_unemployment=None,
    previous_output_gap=0.0,
    interest_rate_pressure=0.0,
    **_legacy_arguments,
):
    """Run the equations once, in the same sequence as a game quarter.

    ``interest_rate_pressure`` retains its historical name for saved-game and
    UI compatibility.  In the new model it is the effective real rate R_t.
    The current policy choice determines r_t, but r_t enters R only next turn;
    consequently it is calculated by the coordinator for history, not used in
    this quarter's IS equation.
    """
    expected_inflation = calculate_expected_inflation(
        previous_inflation, target_inflation, reputation, parameters
    )
    expected_gap = expected_output_gap(previous_output_gap, parameters)
    unconstrained_gap = dynamic_is_equation(
        expected_gap,
        interest_rate_pressure,
        equilibrium_real_rate,
        demand_shock,
        parameters,
    )
    unemployment_intercept = (
        previous_unemployment if natural_unemployment is None else natural_unemployment
    )
    output_gap = apply_output_capacity(
        unconstrained_gap, unemployment_intercept, parameters
    )
    raw_inflation = new_keynesian_phillips_curve(
        expected_inflation, output_gap, inflation_shock, parameters
    )
    inflation = apply_inflation_floor(slow_deflation(raw_inflation, parameters), parameters)
    unemployment = okuns_law(unemployment_intercept, output_gap, parameters)
    output_growth = calculate_output_growth(output_gap, previous_output_gap, parameters)

    return MotionResult(
        inflation=inflation,
        output_growth=output_growth,
        unemployment=unemployment,
        output_gap=output_gap,
        aggregate_demand=unconstrained_gap,
        aggregate_supply=raw_inflation,
        expected_inflation=float(expected_inflation),
    )


# Names retained for small external extensions written against the prior model.
calculate_vertical_supply_output_gap = maximum_output_gap
calculate_vertical_supply_output_growth = maximum_output_gap
