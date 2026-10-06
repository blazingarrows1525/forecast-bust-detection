"""The B2 non-inferiority rule and its validation gates (docs/PREREGISTRATION_B2.md). Pure.

The question is whether rebuilding the 30 analysis-derived inputs from the
operational HRES initial state (with derived TCWV) costs the bust model
material ranking skill. Non-inferiority, not superiority: the candidate need
only stay within a margin of the ERA5-built incumbent.
"""
from __future__ import annotations

import numpy as np

YEARS = (2019, 2020, 2021, 2022)
MARGIN = 0.005                     # AUROC
GATE_YEARS = (2016, 2017)          # training years in every fold; never a test year
INDIA_BOX = (6.0, 38.0, 66.0, 100.0)
#: §4 of the design, fixed before any comparison: kg m-2 and Pearson r
G1 = {"rmse_max": 2.0, "abs_bias_max": 1.5}
G2 = {"r_min": 0.95, "rmse_max": 4.0}


def verdict(lo: float, hi: float, margin: float = MARGIN) -> str:
    """Two-sided 95% interval of candidate - incumbent against -margin."""
    if lo > -margin:
        return "non_inferior"
    if hi < -margin:
        return "inferior"
    return "inconclusive"


VERDICT_TEXT = {
    "non_inferior": "the operational-analysis features are non-inferior to ERA5's "
                    "across 2019–2022 (margin 0.005 AUROC)",
    "inferior": "the operational-analysis features are inferior to ERA5's across "
                "2019–2022 (by more than 0.005 AUROC)",
    "inconclusive": "non-inferiority of the operational-analysis features is not "
                    "established across 2019–2022 (margin 0.005 AUROC)",
}


def field_agreement(derived: np.ndarray, reference: np.ndarray) -> dict:
    """RMSE, mean bias (derived - reference) and Pearson r over finite pairs."""
    d = np.asarray(derived, float).ravel()
    r = np.asarray(reference, float).ravel()
    ok = np.isfinite(d) & np.isfinite(r)
    d, r = d[ok], r[ok]
    diff = d - r
    return {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "bias": float(np.mean(diff)), "r": float(np.corrcoef(d, r)[0, 1])}


def gate_g1(stats: dict) -> bool:
    return stats["rmse"] <= G1["rmse_max"] and abs(stats["bias"]) <= G1["abs_bias_max"]


def gate_g2(stats: dict) -> bool:
    return stats["r"] >= G2["r_min"] and stats["rmse"] <= G2["rmse_max"]
