"""The economy's equations, written in textbook AD--AS order."""

from dataclasses import dataclass

import numpy as np

from parameters import EconomyParameters


@dataclass(frozen=True)
class MotionResult:
    """The solution of this quarter's laws of motion."""

    inflation: float
    output_growth: float
    unemployment: float
    output_gap: float
    aggregate_demand: float
    aggregate_supply: float
    expected_inflation: float = 2.0


class NumericalSolutionError(RuntimeError):
    """Raised when the AD and AS curves do not reach an intersection."""


def calculate_expected_inflation(
    previous_inflation, target_inflation, reputation, parameters
):
    """Form the ex-ante inflation expectation used by supply and real rates."""
    if previous_inflation is None:
        return parameters.expected_inflation
    target = (
        parameters.inflation_target
        if target_inflation is None
        else target_inflation
    )
    alpha = 0.0 if reputation is None else min(1.0, max(0.0, reputation / 10.0))
    return alpha * target + (1.0 - alpha) * previous_inflation


def calculate_interest_rate_pressure(
    real_interest_rates, equilibrium_real_rates, previous_pressure, parameters
):
    """Update smoothed monetary-policy pressure using the two-quarter lag.

    ``z_t = rho * (r_{t-2} - r*_{t-2}) + (1-rho) * z_{t-1}``.
    With fewer than two observations there is no eligible rate gap yet, so the
    previous pressure is carried forward.
    """
    rates = np.asarray(real_interest_rates, dtype=float)
    equilibrium_rates = np.asarray(equilibrium_real_rates, dtype=float)
    if rates.shape != equilibrium_rates.shape:
        raise ValueError(
            "real and equilibrium real interest rate histories must have equal length"
        )

    rho = parameters.interest_rate_pressure_persistence
    if not 0.0 <= rho <= 1.0:
        raise ValueError("interest-rate pressure persistence must be between 0 and 1")
    if rates.size < 2:
        return float(previous_pressure)
    lagged_gap = rates[-2] - equilibrium_rates[-2]
    return float(rho * lagged_gap + (1.0 - rho) * previous_pressure)


def aggregate_demand_curve(
    inflation,
    player_interest_rate,
    equilibrium_real_rate,
    demand_shock,
    parameters,
    *,
    previous_output_gap=0.0,
    demand_shift=None,
    expected_inflation=None,
    interest_rate_pressure=0.0,
):
    """Return the current output gap on the downward-sloping AD curve.

    Quantity theory gives ``g = (m + v) - inflation``.  Nominal-demand growth
    ``m + v`` equals neutral nominal growth plus current and trailing monetary
    policy effects.  The difference between actual and potential growth updates
    the inherited output-level gap.
    """
    expectation = (
        parameters.expected_inflation
        if expected_inflation is None
        else expected_inflation
    )
    shift = parameters.demand_intercept if demand_shift is None else demand_shift

    # Far from equilibrium, households and firms still spend enough to meet
    # basic needs and keep productive capital operating.  Apply that guardrail
    # only to autonomous demand; policy, shocks, and current inflation retain
    # their full marginal effects.
    autonomous_demand_growth = max(
        parameters.minimum_autonomous_demand_growth,
        parameters.potential_growth + expectation + shift,
    )
    nominal_demand_growth = (
        autonomous_demand_growth
        - parameters.demand_interest_rate_pressure * interest_rate_pressure
        + demand_shock
    )
    actual_growth = nominal_demand_growth - inflation
    return previous_output_gap + (
        actual_growth - parameters.potential_growth
    ) / parameters.periods_per_year


def calculate_vertical_supply_output_gap(natural_unemployment, parameters):
    """Convert the AS unemployment floor into a maximum output-level gap."""
    if parameters.okun_coefficient <= 0:
        raise ValueError("okun_coefficient must be positive")
    return (
        natural_unemployment - parameters.vertical_supply_unemployment
    ) / parameters.okun_coefficient


def calculate_vertical_supply_output_growth(natural_unemployment, parameters):
    """Compatibility wrapper returning the capacity output gap, not growth."""
    return calculate_vertical_supply_output_gap(natural_unemployment, parameters)


def aggregate_supply_curve(
    output_gap,
    inflation_shock,
    parameters,
    expected_inflation=None,
    *,
    regime="normal",
):
    """Return inflation on a branch of the aggregate-supply curve.

    The normal Phillips curve is the local approximation around equilibrium.
    Once it reaches zero inflation, downward price stickiness gives the
    deflation branch ten percent of the normal slope.  Both branches meet at
    the zero-inflation kink.
    """
    expectation = (
        parameters.expected_inflation
        if expected_inflation is None
        else expected_inflation
    )
    normal_intercept = expectation + inflation_shock
    if regime == "normal":
        return normal_intercept + parameters.phillips_output_gap * output_gap
    if regime == "deflation":
        if parameters.phillips_output_gap <= 0:
            raise ValueError("phillips_output_gap must be positive")
        zero_inflation_gap = -normal_intercept / parameters.phillips_output_gap
        deflation_slope = (
            parameters.phillips_output_gap
            * parameters.deflation_supply_slope_ratio
        )
        return deflation_slope * (output_gap - zero_inflation_gap)
    raise ValueError(f"unknown aggregate-supply regime: {regime}")


def okuns_law(natural_unemployment, output_gap, parameters):
    """Translate the output-level gap into an unemployment gap."""
    unemployment = natural_unemployment - parameters.okun_coefficient * output_gap
    return min(
        parameters.maximum_unemployment,
        max(parameters.minimum_unemployment, unemployment),
    )


def ad_as_errors(
    candidate_inflation,
    candidate_output_gap,
    player_interest_rate,
    equilibrium_real_rate,
    inflation_shock,
    demand_shock,
    parameters,
    *,
    previous_output_gap=0.0,
    expected_inflation=None,
    demand_shift=None,
    interest_rate_pressure=0.0,
    supply_regime="normal",
):
    """Measure the two errors at a candidate point in (output gap, inflation)."""
    inflation_on_as_curve = aggregate_supply_curve(
        candidate_output_gap,
        inflation_shock,
        parameters,
        expected_inflation=expected_inflation,
        regime=supply_regime,
    )
    output_gap_on_ad_curve = aggregate_demand_curve(
        candidate_inflation,
        player_interest_rate,
        equilibrium_real_rate,
        demand_shock,
        parameters,
        previous_output_gap=previous_output_gap,
        demand_shift=demand_shift,
        expected_inflation=expected_inflation,
        interest_rate_pressure=interest_rate_pressure,
    )
    return np.array(
        [
            candidate_inflation - inflation_on_as_curve,
            candidate_output_gap - output_gap_on_ad_curve,
        ],
        dtype=float,
    )


def numerical_jacobian(error_function, candidate, step_size):
    """Estimate how both equation errors respond to small changes."""
    number_of_equations = len(candidate)
    jacobian = np.zeros((number_of_equations, number_of_equations), dtype=float)
    for variable_index in range(number_of_equations):
        small_change = np.zeros(number_of_equations, dtype=float)
        small_change[variable_index] = step_size
        errors_after_increase = error_function(candidate + small_change)
        errors_after_decrease = error_function(candidate - small_change)
        jacobian[:, variable_index] = (
            errors_after_increase - errors_after_decrease
        ) / (2.0 * step_size)
    return jacobian


def find_curve_intersection(error_function, initial_guess, parameters):
    """Find the AD--AS intersection with Newton's numerical method."""
    candidate = np.array(initial_guess, dtype=float)
    for _iteration in range(parameters.solver_max_iterations):
        current_errors = error_function(candidate)
        if np.max(np.abs(current_errors)) <= parameters.solver_tolerance:
            return candidate
        jacobian = numerical_jacobian(
            error_function, candidate, parameters.solver_step_size
        )
        try:
            newton_step = np.linalg.solve(jacobian, -current_errors)
        except np.linalg.LinAlgError as error:
            raise NumericalSolutionError(
                "AD and AS do not have a numerically identifiable intersection"
            ) from error
        candidate = candidate + newton_step
    raise NumericalSolutionError(
        "AD and AS did not converge within "
        f"{parameters.solver_max_iterations} iterations"
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
    demand_shift=None,
    vertical_supply_output_gap=None,
    vertical_supply_output_growth=None,
    interest_rate_pressure=0.0,
):
    """Solve the guarded AD--AS model in output-gap--inflation space.

    The numerical solver first tries the normal AS branch.  A result below the
    zero-inflation kink is re-solved on the flatter deflation branch; a result
    beyond productive capacity is instead re-solved against vertical AS.
    """
    expected_inflation = calculate_expected_inflation(
        previous_inflation, target_inflation, reputation, parameters
    )
    unemployment_intercept = (
        previous_unemployment if natural_unemployment is None else natural_unemployment
    )
    resolved_demand_shift = (
        parameters.demand_intercept if demand_shift is None else demand_shift
    )

    def errors_at(candidate, supply_regime="normal"):
        return ad_as_errors(
            candidate[0],
            candidate[1],
            player_interest_rate,
            equilibrium_real_rate,
            inflation_shock,
            demand_shock,
            parameters,
            previous_output_gap=previous_output_gap,
            expected_inflation=expected_inflation,
            demand_shift=resolved_demand_shift,
            interest_rate_pressure=interest_rate_pressure,
            supply_regime=supply_regime,
        )

    inflation, output_gap = find_curve_intersection(
        errors_at, [expected_inflation, previous_output_gap], parameters
    )

    capacity = vertical_supply_output_gap
    if capacity is None:
        capacity = vertical_supply_output_growth
    supply_is_vertical = capacity is not None and output_gap > capacity
    supply_regime = "normal"
    if supply_is_vertical:
        # Vertical AS fixes output at capacity, while AD determines inflation.
        def vertical_errors(candidate):
            candidate_inflation, candidate_output_gap = candidate
            demand_gap = aggregate_demand_curve(
                candidate_inflation,
                player_interest_rate,
                equilibrium_real_rate,
                demand_shock,
                parameters,
                previous_output_gap=previous_output_gap,
                demand_shift=resolved_demand_shift,
                expected_inflation=expected_inflation,
                interest_rate_pressure=interest_rate_pressure,
            )
            return np.array(
                [candidate_output_gap - capacity, candidate_output_gap - demand_gap]
            )

        inflation, output_gap = find_curve_intersection(
            vertical_errors, [inflation, capacity], parameters
        )
    elif inflation < 0.0:
        supply_regime = "deflation"
        inflation, output_gap = find_curve_intersection(
            lambda candidate: errors_at(candidate, supply_regime),
            [inflation, output_gap],
            parameters,
        )

    aggregate_supply = aggregate_supply_curve(
        output_gap,
        inflation_shock,
        parameters,
        expected_inflation=expected_inflation,
        regime=supply_regime,
    )
    if supply_is_vertical:
        aggregate_supply = inflation
    aggregate_demand = aggregate_demand_curve(
        inflation,
        player_interest_rate,
        equilibrium_real_rate,
        demand_shock,
        parameters,
        previous_output_gap=previous_output_gap,
        demand_shift=resolved_demand_shift,
        expected_inflation=expected_inflation,
        interest_rate_pressure=interest_rate_pressure,
    )
    output_growth = parameters.potential_growth + parameters.periods_per_year * (
        output_gap - previous_output_gap
    )
    unemployment = okuns_law(unemployment_intercept, output_gap, parameters)

    return MotionResult(
        inflation=float(inflation),
        output_growth=float(output_growth),
        unemployment=float(unemployment),
        output_gap=float(output_gap),
        aggregate_demand=float(aggregate_demand),
        aggregate_supply=float(aggregate_supply),
        expected_inflation=float(expected_inflation),
    )
