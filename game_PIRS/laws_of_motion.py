"""The economy's equations, written in the order used in an undergraduate model."""

from dataclasses import dataclass

import numpy as np

from parameters import EconomyParameters


@dataclass(frozen=True)
class MotionResult:
    """The solution of this quarter's three laws of motion."""

    inflation: float
    output_growth: float
    unemployment: float
    output_gap: float
    aggregate_demand: float
    aggregate_supply: float


class NumericalSolutionError(RuntimeError):
    """Raised when the AD and AS curves do not reach an intersection."""


def aggregate_supply_curve(output_growth, inflation_shock, parameters):
    """Return inflation on the Aggregate Supply (Phillips) Curve.

    Phillips Curve / Aggregate Supply:

        inflation = expected inflation
                    + Phillips coefficient * output gap
                    + inflation shock
    """
    output_gap = output_growth - parameters.potential_growth

    inflation = (
        parameters.expected_inflation
        + parameters.phillips_output_gap * output_gap
        + inflation_shock
    )
    return inflation


def aggregate_demand_curve(
    inflation,
    player_interest_rate,
    equilibrium_real_rate,
    demand_shock,
    parameters,
):
    """Return GDP growth on the Aggregate Demand Curve.

    Aggregate Demand:

        GDP growth = demand intercept
                     + interest sensitivity
                       * (player rate - inflation - equilibrium real rate)
                     + demand shock

    ``player_interest_rate`` is the nominal policy rate chosen by the player.
    """
    real_interest_rate_gap = (
        player_interest_rate - inflation - equilibrium_real_rate
    )

    output_growth = (
        parameters.demand_intercept
        + parameters.demand_real_rate * real_interest_rate_gap
        + demand_shock
    )
    return output_growth


def okuns_law(previous_unemployment, output_growth, parameters):
    """Choose unemployment after AD and AS have determined GDP growth.

    Okun's Law:

        unemployment = last quarter's unemployment
                       - Okun coefficient * output gap
    """
    output_gap = output_growth - parameters.potential_growth
    unemployment = previous_unemployment - parameters.okun_coefficient * output_gap

    bounded_unemployment = min(
        parameters.maximum_unemployment,
        max(parameters.minimum_unemployment, unemployment),
    )
    return bounded_unemployment


def ad_as_errors(
    candidate_inflation,
    candidate_output_growth,
    player_interest_rate,
    equilibrium_real_rate,
    inflation_shock,
    demand_shock,
    parameters,
):
    """Measure how far a candidate point is from lying on both curves.

    The correct solution makes both returned errors equal to zero:

    1. candidate inflation equals inflation on the AS curve;
    2. candidate GDP growth equals GDP growth on the AD curve.
    """
    inflation_on_as_curve = aggregate_supply_curve(
        candidate_output_growth,
        inflation_shock,
        parameters,
    )
    output_growth_on_ad_curve = aggregate_demand_curve(
        candidate_inflation,
        player_interest_rate,
        equilibrium_real_rate,
        demand_shock,
        parameters,
    )

    inflation_error = candidate_inflation - inflation_on_as_curve
    output_growth_error = candidate_output_growth - output_growth_on_ad_curve
    return np.array([inflation_error, output_growth_error], dtype=float)


def numerical_jacobian(error_function, candidate, step_size):
    """Estimate how both equation errors respond to small changes in pi and y.

    This central-difference Jacobian is calculated from the equations themselves.
    Consequently, a future edit to an AD or AS formula does not require anyone to
    derive and update a separate substitution formula or analytical derivative.
    """
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
    """Find the AD-AS intersection with Newton's numerical method.

    Each iteration asks: "What small change to inflation and GDP growth would
    eliminate the current equation errors?" It stops once both errors are below
    the configured tolerance.
    """
    candidate = np.array(initial_guess, dtype=float)

    for _iteration in range(parameters.solver_max_iterations):
        current_errors = error_function(candidate)
        largest_error = np.max(np.abs(current_errors))
        if largest_error <= parameters.solver_tolerance:
            return candidate

        jacobian = numerical_jacobian(
            error_function,
            candidate,
            parameters.solver_step_size,
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
):
    """Numerically solve AD and AS, then use Okun's Law for unemployment."""

    def errors_at(candidate):
        candidate_inflation = candidate[0]
        candidate_output_growth = candidate[1]
        return ad_as_errors(
            candidate_inflation,
            candidate_output_growth,
            player_interest_rate,
            equilibrium_real_rate,
            inflation_shock,
            demand_shock,
            parameters,
        )

    starting_point = [
        parameters.expected_inflation,
        parameters.potential_growth,
    ]
    inflation, output_growth = find_curve_intersection(
        errors_at,
        starting_point,
        parameters,
    )

    # Re-evaluate both named curves at the numerical solution. Keeping these
    # intermediate values visible makes the result easy to inspect and test.
    aggregate_supply = aggregate_supply_curve(
        output_growth,
        inflation_shock,
        parameters,
    )
    aggregate_demand = aggregate_demand_curve(
        inflation,
        player_interest_rate,
        equilibrium_real_rate,
        demand_shock,
        parameters,
    )
    output_gap = output_growth - parameters.potential_growth
    unemployment = okuns_law(
        previous_unemployment,
        output_growth,
        parameters,
    )

    return MotionResult(
        inflation=float(inflation),
        output_growth=float(output_growth),
        unemployment=float(unemployment),
        output_gap=float(output_gap),
        aggregate_demand=float(aggregate_demand),
        aggregate_supply=float(aggregate_supply),
    )
