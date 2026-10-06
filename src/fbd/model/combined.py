"""The served model + ENS combination (D-029, D-030; v0.3.0: D-037). Pure numpy.

Fitted once on 2021, the served base model's calibration year, from the base
model's *uncalibrated* probability and the 50-member ENS spread. Its
coefficients live in data/artifacts/combiner.json, so serving needs no fitting
library and the numbers can be read in review.

Two formats:

- **v1** (store v0.2.0): one logistic combiner on [logit p_raw, log1p spread],
  keys ``logit_p_model``, ``log1p_ens_spread``, ``intercept``.
- **v2** (store v0.3.0): ``{"version": 2, "form": "logit" | "beta", "main": {...},
  "short": {...} | None, "short_leads": [1, 2], ...}``. The beta form uses
  [ln p_raw, ln(1 - p_raw), log1p spread] (D-035); ``short`` is the coefficient
  set fitted on Days 1-2 (D-036), always in the logit form it was tested in.
  Rows at other leads use ``main``.
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


def _short_mask(c: dict, lead, n: int) -> np.ndarray:
    """Rows served by the Day 1-2 set (v2 with a ``short`` set and leads given)."""
    if "version" not in c or c.get("short") is None or lead is None:
        return np.zeros(n, dtype=bool)
    return np.isin(np.broadcast_to(np.asarray(lead), (n,)), c.get("short_leads", []))


def model_term(c: dict, p_raw, lead=None) -> np.ndarray:
    """The base model's share of the served log-odds, per row."""
    p = np.clip(np.asarray(p_raw, dtype=float).ravel(), CLIP, 1 - CLIP)
    if "version" not in c:
        return c["logit_p_model"] * _logit(p)
    short = _short_mask(c, lead, p.size)
    main, out = c["main"], np.empty(p.size)
    m = ~short
    if c["form"] == "beta":
        out[m] = main["ln_p"] * np.log(p[m]) + main["ln_1mp"] * np.log1p(-p[m])
    else:
        out[m] = main["logit_p_model"] * _logit(p[m])
    if short.any():
        out[short] = c["short"]["logit_p_model"] * _logit(p[short])
    return out


def _spread_and_intercept(c: dict, lead, n: int):
    short = _short_mask(c, lead, n)
    main, s = c["main"], c.get("short") or c["main"]
    return (np.where(short, s["log1p_ens_spread"], main["log1p_ens_spread"]),
            np.where(short, s["intercept"], main["intercept"]))


def apply(c: dict, p_raw, spread, lead=None) -> np.ndarray:
    """P(bust) from the model's uncalibrated probability and ENS spread; NaN spread -> NaN."""
    spread = np.asarray(spread, dtype=float)
    p_raw = np.asarray(p_raw, dtype=float)
    if "version" not in c:  # v1, exactly as served by v0.2.0
        z = (c["logit_p_model"] * _logit(p_raw)
             + c["log1p_ens_spread"] * np.log1p(spread) + c["intercept"])
        return 1.0 / (1.0 + np.exp(-z))
    b, a0 = _spread_and_intercept(c, lead, p_raw.size)
    z = model_term(c, p_raw, lead) + b * np.log1p(spread.ravel()) + a0
    return (1.0 / (1.0 + np.exp(-z))).reshape(p_raw.shape)


def contribution(c: dict, spread, lead=None) -> np.ndarray:
    """Log-odds the ensemble adds against the fit year's median spread."""
    spread = np.asarray(spread, dtype=float)
    if "version" not in c:
        return c["log1p_ens_spread"] * (np.log1p(spread) - c["ref_log1p_spread"])
    b, _a0 = _spread_and_intercept(c, lead, spread.size)
    return b * (np.log1p(spread.ravel()) - c["ref_log1p_spread"])


def ens_reason(spread_mm: float, contrib: float) -> str:
    """One reason line, in the same plain register as the model's reason lines."""
    if abs(contrib) < TYPICAL:
        return (f"the 50-member ensemble spread is typical ({spread_mm:.1f} mm/day): "
                "little effect on the bust odds")
    level, verb = ("high", "raises") if contrib > 0 else ("low", "lowers")
    return (f"the 50-member ensemble spread is {level} ({spread_mm:.1f} mm/day): "
            f"{verb} the bust odds {np.exp(abs(contrib)):.1f}x")
