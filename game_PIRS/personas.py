"""Central-bank personas and their policy rules."""

import numpy as np


def draw_persona(random_value=None):
    value = np.random.rand() if random_value is None else random_value
    if value < 0.50:
        return "good"
    if value < 0.75:
        return "hawk"
    if value < 0.95:
        return "dove"
    return "careless"


def taylor_rate(persona, inflation, unemployment, natural_unemployment, natural_rate):
    if persona == "good":
        return natural_rate + inflation + 0.5 * (inflation - 2) + 0.5 * (natural_unemployment - unemployment)
    if persona == "dove":
        return natural_rate - 1 + inflation + 0.1 * (inflation - 4) + 0.9 * (natural_unemployment - unemployment)
    if persona == "hawk":
        return natural_rate + 1 + inflation + 0.9 * (inflation - 1.5) + 0.1 * (natural_unemployment - unemployment)
    return natural_rate - 1 + inflation + 0.05 * (inflation - 6) + 0.95 * (natural_unemployment - 3 - unemployment)


def automated_rate(persona, current_rate, indicators):
    """Move gradually toward a persona's desired policy rate.

    Personas normally hold when the desired rate is within half a percentage
    point, move by a quarter point for moderate gaps, and by half a point for
    gaps greater than one point. Good and hawk policymakers retain the special
    emergency inflation response, while the hawk is excluded from the emergency
    recession response.
    """
    desired = taylor_rate(
        persona,
        indicators.inflation_rate,
        indicators.unemployment_rate,
        indicators.natural_unemployment_rate,
        indicators.real_rate_eq,
    )
    gap = desired - current_rate

    if (
        persona != "hawk"
        and indicators.unemployment_rate > 6
        and indicators.inflation_rate < 1
    ):
        new_rate = 0.0
    elif (
        persona in ("good", "hawk")
        and gap > 1
        and indicators.inflation_rate > 10
        and indicators.unemployment_rate <= 10
    ):
        new_rate = int(indicators.inflation_rate * 1.1 + 5)
    elif gap > 1:
        new_rate = current_rate + 0.5
    elif gap > 0.5:
        new_rate = current_rate + 0.25
    elif gap < -1:
        new_rate = current_rate - 0.5
    elif gap < -0.5:
        new_rate = current_rate - 0.25
    else:
        new_rate = current_rate

    return min(max(round(new_rate * 4) / 4, 0.0), current_rate * 9 + 10)
