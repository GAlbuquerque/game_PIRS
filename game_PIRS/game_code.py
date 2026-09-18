"""Portable, file-free serialization for games in progress."""

from collections import defaultdict
from dataclasses import asdict
import json
import math

from economy import Economy
from history import EconomicHistory, HistoryEntry
from indicators import EconomicIndicators
from parameters import EconomyParameters
from variables import Variables

GAME_CODE_PREFIX = "PIRSG1:"


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"the saved game has an invalid {name}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"the saved game has an invalid {name}")
    return value


def encode_game_code(economy: Economy, game_state: dict) -> str:
    """Return a deterministic JSON code containing a complete game position."""
    parameters = asdict(economy.parameters)
    correlations = economy.parameters.shock_correlations
    parameters["shock_correlations"] = (
        correlations.tolist() if hasattr(correlations, "tolist") else correlations
    )
    payload = {
        "economy": {
            "parameters": parameters,
            "indicators": asdict(economy.indicators),
            "difficulty": economy.difficulty,
            "minimum_interest_rate": economy.minimum_interest_rate,
            "shock_sd_scale": economy.shock_sd_scale,
            "interest_rate": economy.interest_rate,
            "reputation": economy.reputation,
            "expected_inflation": economy.expected_inflation,
            "cb_persona": economy.cb_persona,
            "current_quarter": economy.current_quarter,
            "max_quarters": economy.max_quarters,
            "offset": economy.offset,
            "player_start_turn": economy.player_start_turn,
            "interest_rate_pressure": economy.interest_rate_pressure,
            "player_event_queue": economy.player_event_queue,
            "player_event_last_used": economy.player_event_last_used,
            "player_event_used_quarter": economy.player_event_used_quarter,
            "history": [asdict(entry) for entry in economy.history.entries],
            "event_engine": {
                "effect_queue": [dict(effects) for effects in economy.effect_queue],
                "past_events": economy.past_events,
                "last_event_quarter": economy.last_event_quarter,
            },
        },
        "game": game_state,
    }
    return GAME_CODE_PREFIX + json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def decode_game_code(code: str) -> tuple[Economy, dict]:
    """Validate and restore a game code without executing serialized objects."""
    stripped = code.strip()
    if stripped[: len(GAME_CODE_PREFIX)].upper() != GAME_CODE_PREFIX:
        raise ValueError("this is not a valid PIRSG1 saved-game code")
    try:
        payload = json.loads(stripped[len(GAME_CODE_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("the saved-game code contains invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"economy", "game"}:
        raise ValueError("the saved-game code does not contain the expected data")
    data, game = payload["economy"], payload["game"]
    if not isinstance(data, dict) or not isinstance(game, dict):
        raise ValueError("the saved-game code does not contain the expected data")
    try:
        parameters = EconomyParameters(**data["parameters"])
        indicators = EconomicIndicators(**data["indicators"])
        entries = []
        for entry in data["history"]:
            entry = dict(entry)
            entry["events"] = tuple(entry.get("events", ()))
            entries.append(HistoryEntry(**entry))
        difficulty = data["difficulty"]
        minimum_rate = _finite_number(data["minimum_interest_rate"], "rate floor")
        if difficulty not in {"principles", "senior", "central_banker"}:
            raise ValueError("the saved game has an invalid difficulty")
        if not entries:
            raise ValueError("the saved game has no economic history")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("the saved game"):
            raise
        raise ValueError("the saved-game code contains invalid economic data") from exc

    economy = Economy(initial_state=indicators, difficulty=difficulty,
                      parameters=parameters, minimum_interest_rate=minimum_rate)
    try:
        for name in ("shock_sd_scale", "interest_rate", "reputation",
                     "expected_inflation", "interest_rate_pressure"):
            setattr(economy, name, _finite_number(data[name], name.replace("_", " ")))
        for name in ("current_quarter", "max_quarters", "offset", "player_start_turn"):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"the saved game has an invalid {name.replace('_', ' ')}")
            setattr(economy, name, value)
        economy.cb_persona = str(data["cb_persona"])
        economy.player_event_queue = list(data["player_event_queue"])
        economy.player_event_last_used = dict(data["player_event_last_used"])
        economy.player_event_used_quarter = data["player_event_used_quarter"]
        economy.history = EconomicHistory(entries)
        engine = data["event_engine"]
        queues = engine["effect_queue"]
        if not isinstance(queues, list) or len(queues) != economy.EVENT_HORIZON:
            raise ValueError("the saved game has an invalid event queue")
        economy.event_engine.effect_queue = [
            defaultdict(float, {key: _finite_number(value, "event effect")
                                for key, value in effects.items()})
            for effects in queues
        ]
        economy.past_events = list(engine["past_events"])
        economy.last_event_quarter = int(engine["last_event_quarter"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("the saved game"):
            raise
        raise ValueError("the saved-game code contains invalid game state") from exc

    economy.variables = Variables()
    for entry in entries:
        for name in ("inflation_rate", "unemployment_rate",
                     "natural_unemployment_rate", "interest_rate", "real_interest_rate"):
            economy.variables.update(name, getattr(entry, name))
        economy.variables.update("unemployment_gap",
                                 entry.unemployment_rate - entry.natural_unemployment_rate)
        economy.variables.update("cb_reputation", entry.reputation)
    return economy, game
