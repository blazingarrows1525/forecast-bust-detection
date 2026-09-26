"""The served model + ENS combination (D-029, D-030). Pure numpy.

Fitted once by scripts/fit_combiner.py on 2021, the frozen model's calibration
year, from the frozen model's *uncalibrated* probability and the 50-member ENS
spread. Its coefficients live in data/artifacts/combiner.json, so serving needs
no fitting library and the numbers can be read in review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fbd import config

COMBINER = config.ARTIFACTS / "combiner.json"
CLIP = 1e-6
TYPICAL = 0.1  # |log-odds contribution| below this reads as "typical"


def load(path: Path = COMBINER) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def apply(c: dict, p_raw, spread) -> np.ndarray:
    """P(bust) from the model's uncalibrated probability and ENS spread; NaN spread -> NaN."""
    z = (c["logit_p_model"] * _logit(p_raw)
         + c["log1p_ens_spread"] * np.log1p(np.asarray(spread, dtype=float))
         + c["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def contribution(c: dict, spread) -> np.ndarray:
    """Log-odds the ensemble adds against the fit year's median spread."""
    return c["log1p_ens_spread"] * (np.log1p(np.asarray(spread, dtype=float))
                                    - c["ref_log1p_spread"])


def ens_reason(spread_mm: float, contrib: float) -> str:
    """One reason line, in the same plain register as the TreeSHAP lines."""
    if abs(contrib) < TYPICAL:
        return (f"the 50-member ensemble spread is typical ({spread_mm:.1f} mm/day): "
                "little effect on the bust odds")
    level, verb = ("high", "raises") if contrib > 0 else ("low", "lowers")
    return (f"the 50-member ensemble spread is {level} ({spread_mm:.1f} mm/day): "
            f"{verb} the bust odds {np.exp(abs(contrib)):.1f}x")
