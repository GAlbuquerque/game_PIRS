from dataclasses import dataclass, field
from math import sqrt
from typing import Optional, Sequence


@dataclass
class EndGameContext:
    mandate: str
    initial_inflation: float
    initial_unemployment: float
    dual_unemployment_target: float
    inflation_history: Sequence[float]
    unemployment_history: Sequence[float]
    real_interest_rate_history: Sequence[float]
    term_event_names: Sequence[str] = field(default_factory=tuple)
    inflation_target: float = 2.0
    policy_deviation_history: Sequence[float] = field(default_factory=tuple)
    lower_bound_quarters: int = 0
    # Backward-compatibility field for older call sites that still pass this name.
    current_event_name: Optional[str] = None


def mandate_targets(
    mandate: str, dual_unemployment_target: float, inflation_target: float = 2.0
):
    """Return the configured targets used to evaluate the mandate."""
    return {
        "inflation": float(inflation_target),
        "unemployment": (
            float(dual_unemployment_target) if mandate == "dual_mandate" else None
        ),
    }


def mandate_text(mandate: str, dual_unemployment_target: int) -> str:
    if mandate == "dual_mandate":
        return (
            f"Dual mandate: keep inflation near 2.0% and unemployment near "
            f"{dual_unemployment_target}% "
        )
    return "Inflation target mandate: keep inflation near 2.0%."


def _rms(values: Sequence[float]) -> float:
    return sqrt(sum(float(value) ** 2 for value in values) / len(values)) if values else 0.0


def mandate_loss(
    mandate: str,
    inflation_history: Sequence[float],
    unemployment_history: Sequence[float],
    inflation_target: float,
    unemployment_target: float,
) -> float:
    """Measure distance from the selected mandate, penalizing volatility naturally."""
    inflation_loss = _rms(
        [float(value) - float(inflation_target) for value in inflation_history]
    )
    if mandate != "dual_mandate":
        return inflation_loss
    unemployment_loss = _rms(
        [
            max(0.0, float(value) - float(unemployment_target))
            for value in unemployment_history
        ]
    )
    return sqrt((inflation_loss**2 + unemployment_loss**2) / 2.0)


def taylor_policy_deviations(
    inflation_history: Sequence[float],
    unemployment_history: Sequence[float],
    natural_unemployment_history: Sequence[float],
    equilibrium_real_rate_history: Sequence[float],
    selected_rate_history: Sequence[float],
    inflation_target: float,
    minimum_interest_rate: float,
) -> tuple[list[float], int]:
    """Compare decisions with the feasible balanced Taylor-rule recommendation."""
    deviations = []
    constrained_quarters = 0
    for inflation, unemployment, natural_unemployment, natural_rate, selected_rate in zip(
        inflation_history,
        unemployment_history,
        natural_unemployment_history,
        equilibrium_real_rate_history,
        selected_rate_history,
    ):
        raw_taylor_rate = (
            float(natural_rate)
            + float(inflation)
            + 0.5 * (float(inflation) - float(inflation_target))
            + 0.5 * (float(natural_unemployment) - float(unemployment))
        )
        feasible_taylor_rate = max(float(minimum_interest_rate), raw_taylor_rate)
        if raw_taylor_rate < float(minimum_interest_rate):
            constrained_quarters += 1
        deviations.append(float(selected_rate) - feasible_taylor_rate)
    return deviations, constrained_quarters


def classify_public_view(policy_deviations: Sequence[float]):
    """Classify revealed preferences relative to a balanced Taylor rule."""
    average_deviation = (
        sum(float(value) for value in policy_deviations) / len(policy_deviations)
        if policy_deviations
        else 0.0
    )
    if average_deviation > 1.0:
        return "Hawk", "Bond markets saw you as inflation-first and uncompromising."
    if average_deviation < -4.0:
        return (
            "Careless",
            "Newspapers decry your failed improvisation and the turbulence it generated. "
            "People wonder if you missed your Macroeconomics classes.",
        )
    if average_deviation < -1.0:
        return "Dove", "Labor groups remember you as employment-first and patient on prices."
    return "Balanced", "Economic historians view you as steady under pressure."


def _join_with_and(items: Sequence[str]) -> str:
    vals = [item for item in items if item]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return f"{vals[0]} and {vals[1]}"
    return f"{', '.join(vals[:-1])}, and {vals[-1]}"


def _context_message(events: Sequence[str]) -> str:
    events = list(events)
    if not events:
        return "Your term has ended without a major economic shock."
    if len(events) <= 2:
        return f"Your term has ended after you navigated {_join_with_and(events)}."
    return f"Your turbulent term featured {_join_with_and(events[:3])}."


def _record_message(mandate: str, loss: float) -> str:
    if mandate == "dual_mandate":
        if loss <= 1.0:
            return "You maintained a strong balance between price stability and employment."
        if loss <= 2.0:
            return "A balance between inflation and employment remained within reach, but you never fully secured it."
        return "A durable balance between inflation and employment eluded you."

    if loss <= 1.0:
        return "You kept inflation close to target over the course of the term."
    if loss <= 2.0:
        return "Price stability remained within reach, but you never fully secured it."
    return "Price stability eluded you for much of the term."


def _direction_message(beginning_loss: float, ending_loss: float) -> str:
    improvement = beginning_loss - ending_loss
    if improvement > 0.5:
        return "By the final year, the economy stood closer to the mandate than at the start."
    if improvement < -0.5:
        return "By the final year, the economy had drifted further from the mandate."
    if ending_loss <= 1.0:
        return "That stability endured through the final year."
    return "The economy made little lasting progress between the opening and final years."


def build_end_of_term_message(ctx: EndGameContext) -> str:
    inflation = list(ctx.inflation_history)
    unemployment = list(ctx.unemployment_history)
    whole_loss = mandate_loss(
        ctx.mandate,
        inflation,
        unemployment,
        ctx.inflation_target,
        ctx.dual_unemployment_target,
    )
    beginning_loss = mandate_loss(
        ctx.mandate,
        inflation[:4],
        unemployment[:4],
        ctx.inflation_target,
        ctx.dual_unemployment_target,
    )
    ending_loss = mandate_loss(
        ctx.mandate,
        inflation[-4:],
        unemployment[-4:],
        ctx.inflation_target,
        ctx.dual_unemployment_target,
    )

    label, reputation = classify_public_view(ctx.policy_deviation_history)
    context = _context_message(ctx.term_event_names)
    record = _record_message(ctx.mandate, whole_loss)
    direction = _direction_message(beginning_loss, ending_loss)
    if label == "Balanced" and ctx.lower_bound_quarters >= 4:
        reputation = (
            "Economic historians view you as steady under pressure, even when the "
            "lower bound left little room to maneuver."
        )

    return (
        f"{context} {reputation} They classify you as: {label}\n\n"
        f"{record} {direction}"
    )
