"""Named, deliberately unsimplified macroeconomic laws of motion."""

from dataclasses import dataclass

from parameters import EconomyParameters


@dataclass(frozen=True)
class MotionResult:
    inflation: float
    output_growth: float
    unemployment: float
    output_gap: float
    aggregate_demand: float
    aggregate_supply: float


def solve_ad_as(
    interest_rate,
    equilibrium_real_rate,
    previous_unemployment,
    inflation_shock,
    demand_shock,
    parameters: EconomyParameters,
):
    """Solve Aggregate Demand and the Phillips (Aggregate Supply) curve.

    AS / Phillips curve:
        pi = pi^e + beta_pi * (y - y^p) + epsilon_pi
    AD curve:
        y = beta0 + beta_y * (i_t - pi - r*) + epsilon_y
    Okun's law (evaluated only after AD-AS has been solved):
        u = u[-1] - beta_u * (y - y^p)

    ``i_t`` is the nominal interest rate chosen by the player.  Intermediate
    curve values remain explicit in the return object for teaching and editing.
    """
    p = parameters
    denominator = 1.0 + p.demand_real_rate * p.phillips_output_gap
    if abs(denominator) < 1e-12:
        raise ValueError("AD and AS curves are parallel and cannot be solved")

    # Substitute the complete AS curve into the complete AD curve and solve y.
    output_growth = (
        p.demand_intercept
        + p.demand_real_rate
        * (
            interest_rate
            - equilibrium_real_rate
            - p.expected_inflation
            + p.phillips_output_gap * p.potential_growth
            - inflation_shock
        )
        + demand_shock
    ) / denominator

    output_gap = output_growth - p.potential_growth
    aggregate_supply = (
        p.expected_inflation
        + p.phillips_output_gap * output_gap
        + inflation_shock
    )
    aggregate_demand = (
        p.demand_intercept
        + p.demand_real_rate
        * (interest_rate - aggregate_supply - equilibrium_real_rate)
        + demand_shock
    )

    # Okun's equation selects unemployment after pi and y have been determined.
    unemployment = previous_unemployment - p.okun_coefficient * output_gap
    unemployment = min(
        p.maximum_unemployment,
        max(p.minimum_unemployment, unemployment),
    )
    return MotionResult(
        inflation=aggregate_supply,
        output_growth=output_growth,
        unemployment=unemployment,
        output_gap=output_gap,
        aggregate_demand=aggregate_demand,
        aggregate_supply=aggregate_supply,
    )
