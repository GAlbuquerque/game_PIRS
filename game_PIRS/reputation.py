"""Central-bank reputation rules."""


def update_reputation(current, previous_inflation, inflation, unemployment, real_rate):
    """Return bounded reputation after applying the game's transparent rules."""
    delta = 0.0
    if inflation < 2:
        delta += 0.02
    if inflation < previous_inflation:
        delta += 0.02
    if real_rate > 4 and unemployment > 10:
        delta += 0.10
    if inflation > 6:
        delta -= 0.05
    if previous_inflation > 2 and inflation > previous_inflation:
        delta -= 0.025
    if real_rate < 2 and previous_inflation > 6:
        delta -= 0.05
    return float(min(1.0, max(0.0, current + delta)))
