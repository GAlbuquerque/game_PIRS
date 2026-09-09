from dataclasses import dataclass, field
from math import sqrt
from typing import Optional, Sequence


FAVORABLE_EVENTS = {"Technological Boom"}
ADVERSE_EVENTS = {
    "Financial Crisis",
    "Major Financial Crisis",
    "Pandemic Outbreak",
    "Natural Disaster",
    "Global Supply Shock",
}
MIXED_EVENTS = {"Fiscal Deficit", "Spending Wave", "Fiscal Surplus"}
EVENT_ALIASES = {
    "Financial Crisis": "a financial crisis",
    "Major Financial Crisis": "a major financial crisis",
    "Technological Boom": "a technological boom",
    "Pandemic Outbreak": "a pandemic outbreak",
    "Natural Disaster": "a natural disaster",
    "Global Supply Shock": "a global supply shock",
    "Fiscal Deficit": "a fiscal deficit",
    "Spending Wave": "a spending wave",
    "Fiscal Surplus": "a fiscal surplus",
}
EVENT_DOMINANCE = {
    "Major Financial Crisis": {"Financial Crisis"},
    "Spending Wave": {"Fiscal Deficit"},
}
EVENT_PRIORITY = {
    "Major Financial Crisis": 9,
    "Pandemic Outbreak": 8,
    "Global Supply Shock": 7,
    "Natural Disaster": 6,
    "Financial Crisis": 5,
    "Spending Wave": 4,
    "Technological Boom": 3,
    "Fiscal Deficit": 2,
    "Fiscal Surplus": 1,
}


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
    return (inflation_loss + unemployment_loss) / 2.0


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


def _performance_band(term_loss: float, beginning_loss: float) -> str:
    if term_loss < 2.0 and beginning_loss < 1.0:
        return "strong"
    if term_loss <= 4.0:
        return "mixed"
    return "poor"


def _record_message(
    mandate: str,
    band: str,
    average_inflation: float,
    inflation_target: float,
) -> str:
    if mandate == "dual_mandate":
        if band == "strong":
            return "You maintained a strong balance between price stability and employment."
        if band == "mixed":
            return "A balance between inflation and employment remained within reach, but you never fully secured it."
        return "A durable balance between inflation and employment eluded you."

    if band == "strong":
        return "You kept inflation close to target over the course of the term."
    if average_inflation < inflation_target - 0.1:
        if band == "mixed":
            return "Inflation ran below target overall, though a recovery in prices remained within reach."
        return "Inflation fell well below target, and deflationary pressure overshadowed much of the term."
    if average_inflation > inflation_target + 0.1:
        if band == "mixed":
            return "Price stability remained within reach, but you never fully secured it."
        return "High inflation kept price stability out of reach for much of the term."
    if band == "mixed":
        return "Price stability remained within reach, but you never fully secured it."
    return "Wide swings in inflation kept price stability out of reach."


def _event_classification(event: str) -> str:
    if event in FAVORABLE_EVENTS:
        return "favorable"
    if event in ADVERSE_EVENTS:
        return "adverse"
    if event in MIXED_EVENTS:
        return "mixed"
    return "mixed"


def _simplify_events(events: Sequence[str]) -> list[str]:
    simplified = list(dict.fromkeys(event for event in events if event))
    for stronger, weaker_events in EVENT_DOMINANCE.items():
        if stronger in simplified:
            simplified = [event for event in simplified if event not in weaker_events]
    if len(simplified) > 3:
        simplified = sorted(
            simplified,
            key=lambda event: EVENT_PRIORITY.get(event, 0),
            reverse=True,
        )[:3]
    return simplified


def _event_message(events: Sequence[str]) -> str:
    if not events:
        return "The term passed without a major economic shock."
    aliases = [EVENT_ALIASES.get(event, event) for event in events]
    return f"It was marked by {_join_with_and(aliases)}."


def _performance_context(events: Sequence[str], band: str) -> str:
    if not events:
        if band == "strong":
            return "Helped by that calm backdrop, your results were strong."
        if band == "mixed":
            return "Even with that calm backdrop, your results were mixed."
        return "Even without a major shock, your results were poor."

    event_classes = {_event_classification(event) for event in events}
    all_favorable = event_classes == {"favorable"}
    all_adverse = event_classes == {"adverse"}
    if all_favorable and band == "strong":
        return "Aided by those favorable conditions, your results were strong."
    elif all_favorable:
        return f"Despite those favorable conditions, your results were {band}."
    elif all_adverse and band == "strong":
        return "Despite those shocks, your results were strong."
    elif all_adverse:
        return f"In this difficult scenario, your results were {band}."
    return f"In a term of competing forces, your results were {band}."


def _direction_message(beginning_loss: float, ending_loss: float) -> str:
    improvement = beginning_loss - ending_loss
    if improvement > 0.5:
        return "By the final year, the economy stood closer to the mandate than at the start."
    if improvement < -0.5:
        return "By the final year, the economy had drifted further from the mandate."
    if ending_loss <= 1.0:
        return "That stability endured through the final year."
    return "The economy made little lasting progress between the opening and final years."


def evaluate_end_of_term(ctx: EndGameContext) -> dict:
    """Return the numeric mandate assessment displayed on demand by the UIs."""
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
    inflation_loss = mandate_loss(
        "inflation_target",
        inflation,
        unemployment,
        ctx.inflation_target,
        ctx.dual_unemployment_target,
    )
    unemployment_loss = _rms(
        [max(0.0, value - ctx.dual_unemployment_target) for value in unemployment]
    )
    return {
        "term_loss": whole_loss,
        "beginning_loss": beginning_loss,
        "ending_loss": ending_loss,
        "inflation_loss": inflation_loss,
        "unemployment_loss": unemployment_loss,
        "performance": _performance_band(whole_loss, beginning_loss),
    }


def build_end_of_term_message(ctx: EndGameContext) -> str:
    inflation = list(ctx.inflation_history)
    evaluation = evaluate_end_of_term(ctx)

    label, reputation = classify_public_view(ctx.policy_deviation_history)
    average_inflation = sum(inflation) / len(inflation) if inflation else ctx.inflation_target
    record = _record_message(
        ctx.mandate,
        evaluation["performance"],
        average_inflation,
        ctx.inflation_target,
    )
    events = _simplify_events(ctx.term_event_names)
    event_message = _event_message(events)
    performance_context = _performance_context(events, evaluation["performance"])
    direction = _direction_message(
        evaluation["beginning_loss"], evaluation["ending_loss"]
    )
    if label == "Balanced" and ctx.lower_bound_quarters >= 4:
        reputation = (
            "Economic historians view you as steady under pressure, even when the "
            "lower bound left little room to maneuver."
        )

    return (
        f"Your term has ended. {event_message}\n\n"
        f"{performance_context} {record} {direction}\n\n"
        f"{reputation} They classify you as: {label}"
    )
