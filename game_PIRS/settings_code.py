"""Portable, deterministic serialization for model calibrations."""

import json

from parameters import EconomyParameters


SETTINGS_CODE_PREFIX = "PIRS2:"
MODEL_PARAMETER_ORDER = [
    "interest_rate_pressure_persistence",
    "demand_interest_rate_pressure",
    "demand_intercept",
    "potential_growth",
    "periods_per_year",
    "phillips_output_gap",
    "deflation_supply_slope_ratio",
    "minimum_autonomous_demand_growth",
    "okun_coefficient",
    "vertical_supply_unemployment",
    "minimum_inflation",
    "minimum_unemployment",
    "maximum_unemployment",
    "expected_inflation",
    "inflation_target",
    "event_probability_scale",
    "natural_unemployment_anchor",
    "natural_unemployment_reversion",
    "minimum_natural_unemployment",
    "solver_tolerance",
    "solver_max_iterations",
    "solver_step_size",
    "shock_std_devs",
]


def encode_settings_code(settings: dict) -> str:
    """Return a readable, one-to-one JSON representation of a calibration."""
    defaults = EconomyParameters()
    calibration = {
        name: settings.get(name, getattr(defaults, name))
        for name in MODEL_PARAMETER_ORDER
    }
    return SETTINGS_CODE_PREFIX + json.dumps(
        calibration,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_settings(settings: object) -> dict:
    if not isinstance(settings, dict) or set(settings) != set(MODEL_PARAMETER_ORDER):
        raise ValueError("the settings code does not contain the expected parameters")

    integer_fields = {"periods_per_year", "solver_max_iterations"}
    for name in MODEL_PARAMETER_ORDER[:-1]:
        try:
            settings[name] = (
                int(settings[name]) if name in integer_fields else float(settings[name])
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("the settings code has a non-numeric parameter") from exc

    shock_values = settings["shock_std_devs"]
    if not isinstance(shock_values, (list, tuple)) or len(shock_values) != 4:
        raise ValueError("the settings code has invalid shock parameters")
    settings["shock_std_devs"] = tuple(float(value) for value in shock_values)
    EconomyParameters(**settings)
    return settings


def decode_settings_code(code: str) -> dict:
    """Decode a readable PIRS2 calibration code."""
    stripped = code.strip()
    if stripped[:len(SETTINGS_CODE_PREFIX)].upper() != SETTINGS_CODE_PREFIX:
        raise ValueError("this is not a valid PIRS2 settings code")
    try:
        settings = json.loads(stripped[len(SETTINGS_CODE_PREFIX):])
    except json.JSONDecodeError as exc:
        raise ValueError("the settings code contains invalid JSON") from exc
    return _validate_settings(settings)
