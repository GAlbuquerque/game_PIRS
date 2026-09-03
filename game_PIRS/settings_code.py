"""Portable, deterministic serialization for model calibrations."""

import json

from parameters import EconomyParameters


SETTINGS_CODE_PREFIX = "PIRS2:"
MODEL_PARAMETER_ORDER = [
    "interest_rate_pressure_persistence",
    "output_gap_expectation_persistence",
    "intertemporal_elasticity_inverse",
    "inflation_expectation_discount",
    "phillips_output_gap",
    "deflation_adjustment_ratio",
    "okun_coefficient",
    "minimum_inflation",
    "minimum_unemployment",
    "expected_inflation",
    "inflation_target",
    "reputation_expectation_coefficient",
    "unemployment_target",
    "event_probability_scale",
    "natural_unemployment_anchor",
    "natural_unemployment_reversion",
    "minimum_natural_unemployment",
    "equilibrium_real_rate_anchor",
    "equilibrium_real_rate_reversion",
    "shock_std_devs",
]

# Codes produced immediately before the September 2026 calibration included a
# separate negative-gap slope and omitted the equilibrium-rate controls.
RECENT_MODEL_PARAMETER_ORDER = (
    (set(MODEL_PARAMETER_ORDER) - {
        "equilibrium_real_rate_anchor", "equilibrium_real_rate_reversion"
    })
    | {"negative_gap_slope_ratio"}
)

# PIRS2 codes shared before the new model did not contain the four parameters
# above and did contain the old vertical-AS setting.  Accept that exact schema
# so existing links remain usable, then fill the new coefficients with defaults.
LEGACY_MODEL_PARAMETER_ORDER = [
    "interest_rate_pressure_persistence",
    "demand_interest_rate_pressure",
    "output_gap_expectation_persistence",
    "intertemporal_elasticity_inverse",
    "potential_growth",
    "periods_per_year",
    "inflation_expectation_discount",
    "phillips_output_gap",
    "negative_gap_slope_ratio",
    "deflation_supply_slope_ratio",
    "okun_coefficient",
    "minimum_inflation",
    "minimum_unemployment",
    "maximum_unemployment",
    "expected_inflation",
    "inflation_target",
    "reputation_expectation_coefficient",
    "unemployment_target",
    "event_probability_scale",
    "natural_unemployment_anchor",
    "natural_unemployment_reversion",
    "minimum_natural_unemployment",
    "solver_tolerance",
    "solver_max_iterations",
    "solver_step_size",
    "shock_std_devs",
]

# PIRS2 codes shared before the new model did not contain the four parameters
# above and did contain the old vertical-AS setting.  Accept that exact schema
# so existing links remain usable, then fill the new coefficients with defaults.
LEGACY_MODEL_PARAMETER_ORDER = [
    "interest_rate_pressure_persistence",
    "demand_interest_rate_pressure",
    "potential_growth",
    "periods_per_year",
    "phillips_output_gap",
    "deflation_supply_slope_ratio",
    "okun_coefficient",
    "vertical_supply_unemployment",
    "minimum_inflation",
    "minimum_unemployment",
    "maximum_unemployment",
    "expected_inflation",
    "inflation_target",
    "reputation_expectation_coefficient",
    "unemployment_target",
    "event_probability_scale",
    "natural_unemployment_anchor",
    "natural_unemployment_reversion",
    "minimum_natural_unemployment",
    "solver_tolerance",
    "solver_max_iterations",
    "solver_step_size",
    "shock_std_devs",
]

OBSOLETE_PARAMETER_NAMES = {
    "demand_interest_rate_pressure",
    "potential_growth",
    "periods_per_year",
    "vertical_supply_unemployment",
    "solver_tolerance",
    "solver_max_iterations",
    "solver_step_size",
    "deflation_supply_slope_ratio",
    "maximum_unemployment",
}

PREVIOUS_MODEL_PARAMETER_ORDER = (
    (set(LEGACY_MODEL_PARAMETER_ORDER) - {"vertical_supply_unemployment"})
    | {
        "output_gap_expectation_persistence",
        "intertemporal_elasticity_inverse",
        "inflation_expectation_discount",
        "negative_gap_slope_ratio",
    }
)


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
    if not isinstance(settings, dict):
        raise ValueError("the settings code does not contain the expected parameters")

    supplied_names = set(settings)
    current_names = set(MODEL_PARAMETER_ORDER)
    legacy_names = set(LEGACY_MODEL_PARAMETER_ORDER)
    previous_names = set(PREVIOUS_MODEL_PARAMETER_ORDER)
    recent_names = set(RECENT_MODEL_PARAMETER_ORDER)
    if supplied_names in (legacy_names, previous_names, recent_names):
        defaults = EconomyParameters()
        old_deflation_ratio = settings.get("deflation_supply_slope_ratio")
        for name in OBSOLETE_PARAMETER_NAMES:
            settings.pop(name, None)
        settings.pop("negative_gap_slope_ratio", None)
        for name in current_names - supplied_names:
            settings[name] = getattr(defaults, name)
        if old_deflation_ratio is not None:
            settings["deflation_adjustment_ratio"] = old_deflation_ratio
    elif supplied_names != current_names:
        raise ValueError("the settings code does not contain the expected parameters")

    for name in MODEL_PARAMETER_ORDER[:-1]:
        if name == "unemployment_target" and settings[name] is None:
            continue
        try:
            settings[name] = float(settings[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("the settings code has a non-numeric parameter") from exc

    shock_values = settings["shock_std_devs"]
    if not isinstance(shock_values, (list, tuple)) or len(shock_values) != 4:
        raise ValueError("the settings code has invalid shock parameters")
    settings["shock_std_devs"] = tuple(float(value) for value in shock_values)
    nonnegative = {
        "inflation_target",
        "unemployment_target",
        "phillips_output_gap",
        "minimum_unemployment",
        "event_probability_scale",
        "natural_unemployment_anchor",
        "minimum_natural_unemployment",
    }
    for name in nonnegative:
        if settings[name] is not None and settings[name] < 0:
            raise ValueError(f"{name.replace('_', ' ')} cannot be negative")
    if any(value < 0 for value in settings["shock_std_devs"]):
        raise ValueError("shock standard deviations cannot be negative")
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
