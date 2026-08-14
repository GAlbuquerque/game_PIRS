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
    """Choose the persona's quarter-point policy rate, bounded at zero."""
    desired = taylor_rate(
        persona,
        indicators.inflation_rate,
        indicators.unemployment_rate,
        indicators.natural_unemployment_rate,
        indicators.real_rate_eq,
    )
    if indicators.unemployment_rate > 6 and indicators.inflation_rate < 1:
        desired = 0.0
    return min(max(round(desired * 4) / 4, 0.0), current_rate * 9 + 10)
