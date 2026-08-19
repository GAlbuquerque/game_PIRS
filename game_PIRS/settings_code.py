"""Portable, deterministic serialization for model calibrations."""

import base64
import binascii
import json
import zlib

from parameters import EconomyParameters


SETTINGS_CODE_PREFIX = "PIRS2:"
LEGACY_SETTINGS_CODE_PREFIX = "PIRS1"
MODEL_PARAMETER_ORDER = [
    "demand_real_rate",
    "demand_intercept",
    "demand_intercept_weight_10",
    "demand_intercept_weight_20",
    "potential_growth",
    "periods_per_year",
    "phillips_output_gap",
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


def _decode_legacy_settings_code(code: str) -> dict:
    """Decode the compressed positional PIRS1 format for backward compatibility."""
    compact = "".join(code.upper().split()).replace("-", "")
    encoded = compact[len(LEGACY_SETTINGS_CODE_PREFIX):]
    try:
        padding = "=" * (-len(encoded) % 8)
        packed = base64.b32decode(encoded + padding, casefold=True)
        compressed, checksum = packed[:-4], packed[-4:]
        if len(checksum) != 4 or zlib.crc32(compressed).to_bytes(4, "big") != checksum:
            raise ValueError("the settings code is incomplete or mistyped")
        payload = zlib.decompress(compressed)
        values = json.loads(payload)
    except (ValueError, TypeError, binascii.Error, json.JSONDecodeError, zlib.error) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("the settings code"):
            raise
        raise ValueError("the settings code is incomplete or mistyped") from exc
    if not isinstance(values, list) or len(values) != len(MODEL_PARAMETER_ORDER):
        raise ValueError("the settings code uses an unsupported format")
    return _validate_settings(dict(zip(MODEL_PARAMETER_ORDER, values)))


def decode_settings_code(code: str) -> dict:
    """Decode a readable PIRS2 code or a legacy compressed PIRS1 code."""
    stripped = code.strip()
    if stripped[:len(SETTINGS_CODE_PREFIX)].upper() == SETTINGS_CODE_PREFIX:
        try:
            settings = json.loads(stripped[len(SETTINGS_CODE_PREFIX):])
        except json.JSONDecodeError as exc:
            raise ValueError("the settings code contains invalid JSON") from exc
        return _validate_settings(settings)

    compact = "".join(stripped.upper().split()).replace("-", "")
    if compact.startswith(LEGACY_SETTINGS_CODE_PREFIX) and len(compact) <= 5000:
        return _decode_legacy_settings_code(stripped)
    raise ValueError("this is not a valid PIRS settings code")
