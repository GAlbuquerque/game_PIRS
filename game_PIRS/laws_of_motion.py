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


class NumericalSolutionError(RuntimeError):
    """Raised when the AD and AS curves do not reach an intersection."""


def calculate_demand_shift(real_interest_rates, equilibrium_real_rates, parameters):
    """Return the effect of trailing real-rate gaps on nominal-demand growth."""
    rates = np.asarray(real_interest_rates, dtype=float)
    equilibrium_rates = np.asarray(equilibrium_real_rates, dtype=float)
    if rates.size == 0:
        raise ValueError("at least one historical real interest rate is required")
    if rates.shape != equilibrium_rates.shape:
        raise ValueError(
            "real and equilibrium real interest rate histories must have equal length"
        )

    rate_gaps = rates - equilibrium_rates
    average_10 = float(np.mean(rate_gaps[-10:]))
    average_20 = float(np.mean(rate_gaps[-20:]))
    return (
        parameters.demand_intercept_weight_10 * average_10
        + parameters.demand_intercept_weight_20 * average_20
    )


def calculate_demand_intercept(
    real_interest_rates, equilibrium_real_rates, parameters
):
    """Compatibility name for the historical nominal-demand-growth shift."""
    return calculate_demand_shift(
        real_interest_rates, equilibrium_real_rates, parameters
    )


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
    output_gap, inflation_shock, parameters, expected_inflation=None
):
    """Return inflation on the expectations-augmented aggregate-supply curve.

    ``inflation = expected inflation + gamma * output gap + supply shock``
    """
    expectation = (
        parameters.expected_inflation
        if expected_inflation is None
        else expected_inflation
    )
    return expectation + parameters.phillips_output_gap * output_gap + inflation_shock


def aggregate_demand_curve(
    inflation,
    player_interest_rate,
    equilibrium_real_rate,
    demand_shock,
    parameters,
    *,
    previous_output_gap=0.0,
    demand_shift=None,
    demand_intercept=None,
    expected_inflation=None,
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
    if demand_intercept is not None:
        shift = demand_intercept

    # Current and historical policy stances both use realized inflation.
    current_real_rate_gap = (
        player_interest_rate - inflation - equilibrium_real_rate
    )
    nominal_demand_growth = (
        parameters.potential_growth
        + expectation
        + shift
        + parameters.demand_real_rate * current_real_rate_gap
        + demand_shock
    )
    actual_growth = nominal_demand_growth - inflation
    return previous_output_gap + (
        actual_growth - parameters.potential_growth
    ) / parameters.periods_per_year


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
    demand_intercept=None,
):
    """Measure the two errors at a candidate point in (output gap, inflation)."""
    inflation_on_as_curve = aggregate_supply_curve(
        candidate_output_gap,
        inflation_shock,
        parameters,
        expected_inflation=expected_inflation,
    )
    output_gap_on_ad_curve = aggregate_demand_curve(
        candidate_inflation,
        player_interest_rate,
        equilibrium_real_rate,
        demand_shock,
        parameters,
        previous_output_gap=previous_output_gap,
        demand_shift=demand_shift,
        demand_intercept=demand_intercept,
        expected_inflation=expected_inflation,
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
    demand_intercept=None,
    vertical_supply_output_gap=None,
    vertical_supply_output_growth=None,
):
    """Solve textbook AD and AS in output-gap--inflation space."""
    alpha = 0.0 if reputation is None else min(1.0, max(0.0, reputation / 4.0))
    if previous_inflation is None:
        expected_inflation = parameters.expected_inflation
    else:
        target = (
            parameters.inflation_target
            if target_inflation is None
            else target_inflation
        )
        expected_inflation = alpha * target + (1.0 - alpha) * previous_inflation
    unemployment_intercept = (
        previous_unemployment if natural_unemployment is None else natural_unemployment
    )

    def errors_at(candidate):
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
            demand_shift=demand_shift,
            demand_intercept=demand_intercept,
        )

    inflation, output_gap = find_curve_intersection(
        errors_at, [expected_inflation, previous_output_gap], parameters
    )

    capacity = vertical_supply_output_gap
    if capacity is None:
        capacity = vertical_supply_output_growth
    supply_is_vertical = capacity is not None and output_gap > capacity
    if supply_is_vertical:
        output_gap = float(capacity)
        expectation = expected_inflation
        shift = parameters.demand_intercept if demand_shift is None else demand_shift
        if demand_intercept is not None:
            shift = demand_intercept
        inflation_coefficient = 1.0 + parameters.demand_real_rate
        if inflation_coefficient == 0:
            raise NumericalSolutionError(
                "vertical AS requires demand_real_rate to differ from -1"
            )
        inflation = (
            expectation
            + shift
            + parameters.demand_real_rate
            * (player_interest_rate - equilibrium_real_rate)
            + demand_shock
            - parameters.periods_per_year * (output_gap - previous_output_gap)
        ) / inflation_coefficient

    aggregate_supply = aggregate_supply_curve(
        output_gap,
        inflation_shock,
        parameters,
        expected_inflation=expected_inflation,
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
        demand_shift=demand_shift,
        demand_intercept=demand_intercept,
        expected_inflation=expected_inflation,
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
    )
