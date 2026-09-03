"""Portable, JSON-only save codes for an in-progress game."""

from dataclasses import asdict
import json
import math

from economy import Economy
from history import EconomicHistory, HistoryEntry
from indicators import EconomicIndicators
from parameters import EconomyParameters
from settings_code import MODEL_PARAMETER_ORDER
from variables import Variables


GAME_CODE_PREFIX = "PIRS-GAME1:"
SESSION_FIELDS = (
    "news_log",
    "game_over",
    "player_turn",
    "in_term_quarter",
    "term_start_idx",
    "initial_inflation",
    "initial_unemployment",
    "difficulty",
    "scenario_name",
    "mandate",
    "dual_unemployment_target",
    "inflation_target",
    "end_message",
    "graph_window_mode",
    "graph_split_mode",
    "show_targets_on_graph",
    "end_summary",
    "show_end_dialog",
    "latest_fired",
    "minimum_interest_rate",
)


def _economy_to_dict(economy: Economy) -> dict:
    return {
        "parameters": {
            name: getattr(economy.parameters, name) for name in MODEL_PARAMETER_ORDER
        },
        "indicators": asdict(economy.indicators),
        "interest_rate": economy.interest_rate,
        "reputation": economy.reputation,
        "expected_inflation": economy.expected_inflation,
        "cb_persona": economy.cb_persona,
        "current_quarter": economy.current_quarter,
        "max_quarters": economy.max_quarters,
        "offset": economy.offset,
        "player_start_turn": economy.player_start_turn,
        "interest_rate_pressure": economy.interest_rate_pressure,
        "minimum_interest_rate": economy.minimum_interest_rate,
        "player_event_queue": economy.player_event_queue,
        "player_event_last_used": economy.player_event_last_used,
        "player_event_used_quarter": economy.player_event_used_quarter,
        "history": [asdict(entry) for entry in economy.history.entries],
        "event_engine": {
            "effect_queue": [dict(effects) for effects in economy.effect_queue],
            "past_events": economy.past_events,
            "last_event_quarter": economy.last_event_quarter,
        },
    }


def encode_game_code(economy: Economy, session_state) -> str:
    """Encode the live economy and game UI state as a portable save code."""
    def state_value(name):
        try:
            return session_state[name]
        except KeyError:
            return None

    payload = {
        "economy": _economy_to_dict(economy),
        "session": {
            name: state_value(name) for name in SESSION_FIELDS
        },
    }
    payload["session"]["minimum_interest_rate"] = economy.minimum_interest_rate
    return GAME_CODE_PREFIX + json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"the save code has an invalid {label}")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"the save code has an invalid {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"the save code has an invalid {label}")
    return number


def _restore_variables(economy: Economy) -> None:
    economy.variables = Variables()
    for entry in economy.history.entries:
        values = {
            "inflation_rate": entry.inflation_rate,
            "unemployment_rate": entry.unemployment_rate,
            "natural_unemployment_rate": entry.natural_unemployment_rate,
            "interest_rate": entry.interest_rate,
            "real_interest_rate": entry.real_interest_rate,
            "unemployment_gap": (
                entry.unemployment_rate - entry.natural_unemployment_rate
            ),
            "cb_reputation": entry.reputation,
        }
        for name, value in values.items():
            economy.variables.update(name, value)


def _economy_from_dict(saved: object) -> Economy:
    if not isinstance(saved, dict):
        raise ValueError("the save code does not contain an economy")
    try:
        parameters_data = saved["parameters"]
        if set(parameters_data) != set(MODEL_PARAMETER_ORDER):
            raise ValueError("the save code has incompatible model parameters")
        parameters = EconomyParameters(**parameters_data)
        indicators = EconomicIndicators(**saved["indicators"])
        difficulty = saved["difficulty"] if "difficulty" in saved else None
        # Difficulty belongs to the session in version 1; infer it below from the
        # saved event engine only when reading hand-authored data.
        difficulty = difficulty or "central_banker"
        economy = Economy(
            initial_state=indicators,
            difficulty=difficulty,
            parameters=parameters,
            minimum_interest_rate=saved["minimum_interest_rate"],
        )
        economy.interest_rate = _finite_number(saved["interest_rate"], "interest rate")
        economy.reputation = _finite_number(saved["reputation"], "reputation")
        economy.expected_inflation = _finite_number(
            saved["expected_inflation"], "expected inflation"
        )
        economy.cb_persona = str(saved["cb_persona"])
        for name in ("current_quarter", "max_quarters", "offset", "player_start_turn"):
            setattr(economy, name, int(saved[name]))
        economy.interest_rate_pressure = _finite_number(
            saved["interest_rate_pressure"], "interest-rate pressure"
        )
        economy.player_event_queue = list(saved["player_event_queue"])
        economy.player_event_last_used = {
            str(name): int(quarter)
            for name, quarter in saved["player_event_last_used"].items()
        }
        used_quarter = saved["player_event_used_quarter"]
        economy.player_event_used_quarter = (
            None if used_quarter is None else int(used_quarter)
        )
        history_entries = []
        for entry in saved["history"]:
            entry = dict(entry)
            entry["events"] = tuple(entry.get("events", ()))
            history_entries.append(HistoryEntry(**entry))
        economy.history = EconomicHistory(history_entries)
        engine = saved["event_engine"]
        if len(engine["effect_queue"]) != economy.event_engine.horizon:
            raise ValueError("the save code has an invalid event queue")
        from collections import defaultdict
        economy.event_engine.effect_queue = [
            defaultdict(float, effects) for effects in engine["effect_queue"]
        ]
        economy.past_events = list(engine["past_events"])
        economy.last_event_quarter = int(engine["last_event_quarter"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("the save code"):
            raise
        raise ValueError("the save code is missing or contains invalid game data") from exc
    if not economy.history.entries:
        raise ValueError("the save code has no economic history")
    _restore_variables(economy)
    return economy


def decode_game_code(code: str) -> tuple[Economy, dict]:
    """Validate a game code and return its reconstructed economy and UI state."""
    stripped = code.strip()
    if stripped[: len(GAME_CODE_PREFIX)].upper() != GAME_CODE_PREFIX:
        raise ValueError("this is not a valid PIRS-GAME1 save code")
    try:
        payload = json.loads(stripped[len(GAME_CODE_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("the save code contains invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"economy", "session"}:
        raise ValueError("the save code does not contain the expected game data")
    session = payload["session"]
    if not isinstance(session, dict) or set(session) != set(SESSION_FIELDS):
        raise ValueError("the save code does not contain the expected session data")
    economy = _economy_from_dict(payload["economy"])
    economy.set_difficulty(str(session["difficulty"]))
    return economy, session
