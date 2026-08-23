"""Physical-range gate: refuse to score implausible input rather than guess.

The principle is the one the OOD detector already applies to *atmospheric
states*, pushed one layer earlier to catch *impossible numbers*: a 9,999 mm
daily rainfall or a negative humidity is not a rare event the model should
extrapolate over, it is broken input, and scoring it would produce a confident
number with nothing behind it.

Ranges come from ``config.PHYSICAL_RANGES``, which already existed and is
already cited by the ingest gates. They are deliberately *not* redefined here:
one project, one set of physical bounds, auditable from one file (config.py is
the "every magic number lives here" file by design).

Units matter and are the easy mistake. config.PHYSICAL_RANGES is expressed in
the units the raw NWP fields arrive in -- pressure in Pa, temperature in K --
whereas the modelling frame carries some fields in operational units. The
mapping below is explicit about which is which rather than assuming.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from fbd import config


class InputValidationError(ValueError):
    """Raised when a row cannot be scored because a value is unphysical."""


# Modelling-frame column -> (range key in config.PHYSICAL_RANGES, scale factor
# applied to the column before comparison).
#
# The scale factor is what makes reusing config's raw-unit bounds safe: `mslp`
# is carried in hPa in the dataset but bounded in Pa in config, so it is
# multiplied by 100 before the comparison rather than having a second, silently
# diverging pair of numbers written down here.
COLUMN_RANGES: dict[str, tuple[str, float]] = {
    "fcst_rain_mm": ("rainfall_mm", 1.0),
    "lagged_mean": ("rainfall_mm", 1.0),
    "fcst_prev_run": ("rainfall_mm", 1.0),
    "clim_obs_mean": ("rainfall_mm", 1.0),
    "tcwv": ("tcwv_kgm2", 1.0),
    "z500": ("z500_m2s2", 1.0),
    "mslp": ("msl_pa", 100.0),
    "u850": ("wind_ms", 1.0),
    "v850": ("wind_ms", 1.0),
}

# Fields that must never be negative, whatever else is true of them.
NON_NEGATIVE = ("fcst_rain_mm", "lagged_mean", "lagged_spread", "tcwv", "clim_obs_mean")


@dataclass
class ValidationReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    checked: int = 0

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise InputValidationError("; ".join(self.failures))


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def validate_row(row: dict, required: tuple[str, ...] = ("fcst_rain_mm",)) -> ValidationReport:
    """Check one prediction row.

    Missing optional features are tolerated -- the model handles NaN natively
    and the dropout stress test shows it degrades gracefully (LOGIC.md 12).
    What is *not* tolerated is a value that is present and impossible, because
    that is the case where degradation is silent.
    """
    failures: list[str] = []
    checked = 0

    for name in required:
        if name not in row or _is_missing(row.get(name)):
            failures.append(f"{name} is missing or NaN, and is required to score a row")

    for column, (range_key, scale) in COLUMN_RANGES.items():
        if column not in row:
            continue
        raw = row[column]
        if _is_missing(raw):
            continue  # absent is handled above / tolerated
        checked += 1
        lo, hi = config.PHYSICAL_RANGES[range_key]
        value = float(raw) * scale
        if not (lo <= value <= hi):
            failures.append(
                f"{column}={raw} is outside the physical range for {range_key} "
                f"[{lo}, {hi}] (compared as {value:g})"
            )

    for column in NON_NEGATIVE:
        if column in row and not _is_missing(row[column]) and float(row[column]) < 0:
            checked += 1
            failures.append(f"{column}={row[column]} is negative, which is unphysical")

    return ValidationReport(ok=not failures, failures=failures, checked=checked)


def validate_frame(frame, required: tuple[str, ...] = ("fcst_rain_mm",)) -> ValidationReport:
    """Vectorised gate for a batch. Reports the first few offending columns."""
    failures: list[str] = []
    checked = 0

    for name in required:
        if name not in frame.columns:
            failures.append(f"required column {name} absent from the batch")

    for column, (range_key, scale) in COLUMN_RANGES.items():
        if column not in frame.columns:
            continue
        checked += 1
        lo, hi = config.PHYSICAL_RANGES[range_key]
        scaled = frame[column].astype(float) * scale
        bad = scaled.notna() & ((scaled < lo) | (scaled > hi))
        if bad.any():
            failures.append(
                f"{column}: {int(bad.sum())} row(s) outside [{lo}, {hi}] for {range_key}"
            )

    return ValidationReport(ok=not failures, failures=failures, checked=checked)
