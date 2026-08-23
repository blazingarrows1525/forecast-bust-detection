"""Verification metrics.  Defined before any model exists, on purpose.

LOGIC.md sec 8.2 is explicit: raw accuracy is banned, because with a ~4% bust
rate a model that always says "no bust" scores 96%.  Everything here is either
threshold-free (AUROC, Brier) or explicitly cost-aware.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from fbd import config


def auroc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))


def brier_skill_score(y_true, y_prob, reference: float | None = None) -> float:
    """BSS against a constant-climatology forecast.  >0 means better than climatology."""
    y_true = np.asarray(y_true, dtype=float)
    base = float(y_true.mean()) if reference is None else reference
    bs = brier(y_true, y_prob)
    bs_ref = float(np.mean((base - y_true) ** 2))
    return float(1.0 - bs / bs_ref) if bs_ref > 0 else float("nan")


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Observed frequency vs forecast probability -- the calibration contract.

    Uses equal-count bins rather than equal-width: with rare events most
    predictions crowd into [0, 0.1] and equal-width bins leave the upper bins
    almost empty, producing a diagram that looks dramatic and means nothing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob)
    yt, yp = y_true[order], y_prob[order]
    bins = np.array_split(np.arange(len(yp)), n_bins)
    rows = []
    for b in bins:
        if len(b) == 0:
            continue
        rows.append(
            {
                "n": len(b),
                "mean_predicted": float(yp[b].mean()),
                "observed_frequency": float(yt[b].mean()),
                "p_lo": float(yp[b].min()),
                "p_hi": float(yp[b].max()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    rc = reliability_curve(y_true, y_prob, n_bins)
    w = rc.n / rc.n.sum()
    return float((w * (rc.mean_predicted - rc.observed_frequency).abs()).sum())


def decision_cost(
    y_true,
    y_prob,
    threshold: float,
    cost_miss: float | None = None,
    cost_false_alarm: float | None = None,
) -> dict:
    """Cost of acting on the flags at a given threshold.

    Encodes the asymmetry LOGIC.md sec 8.4 demands: a missed bust is far worse
    than a false low-confidence flag, because a false flag costs a forecaster
    ten minutes and a missed bust costs lives.
    """
    cost_miss = config.COST_MISSED_BUST if cost_miss is None else cost_miss
    cost_false_alarm = (
        config.COST_FALSE_ALARM if cost_false_alarm is None else cost_false_alarm
    )
    y_true = np.asarray(y_true, dtype=int)
    flag = np.asarray(y_prob, dtype=float) >= threshold

    tp = int(np.sum(flag & (y_true == 1)))
    fp = int(np.sum(flag & (y_true == 0)))
    fn = int(np.sum(~flag & (y_true == 1)))
    tn = int(np.sum(~flag & (y_true == 0)))
    cost = cost_miss * fn + cost_false_alarm * fp
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost": float(cost),
        "cost_per_1000_rows": float(1000.0 * cost / max(len(y_true), 1)),
        "pod": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "far": float(fp / (tp + fp)) if (tp + fp) else float("nan"),
        "csi": float(tp / (tp + fp + fn)) if (tp + fp + fn) else float("nan"),
    }


def best_threshold(y_true, y_prob, grid: np.ndarray | None = None) -> float:
    """Threshold minimising the asymmetric decision cost."""
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    costs = [decision_cost(y_true, y_prob, t)["cost"] for t in grid]
    return float(grid[int(np.argmin(costs))])


def potential_economic_value(y_true, y_prob, threshold: float) -> float:
    """Value relative to the better of always-flag / never-flag (Richardson 2000).

    1.0 = a perfect forecast, 0.0 = no better than the best trivial strategy,
    negative = actively harmful.  This is the number that answers "did predicting
    the bust actually change a decision for the better?"
    """
    y_true = np.asarray(y_true, dtype=int)
    d = decision_cost(y_true, y_prob, threshold)
    cm, cf = config.COST_MISSED_BUST, config.COST_FALSE_ALARM
    n = len(y_true)
    base_rate = y_true.mean()

    cost_model = d["cost"] / n
    cost_never = cm * base_rate               # never flag -> every bust missed
    cost_always = cf * (1.0 - base_rate)      # always flag -> every non-bust a false alarm
    cost_ref = min(cost_never, cost_always)
    cost_perfect = 0.0
    if cost_ref - cost_perfect == 0:
        return float("nan")
    return float((cost_ref - cost_model) / (cost_ref - cost_perfect))


def evaluate(y_true, y_prob, threshold: float | None = None, label: str = "") -> dict:
    """The standard bundle reported for every model and baseline."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thr = best_threshold(y_true, y_prob) if threshold is None else threshold
    d = decision_cost(y_true, y_prob, thr)
    return {
        "model": label,
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "auroc": auroc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "bss": brier_skill_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "threshold": thr,
        "pod": d["pod"],
        "far": d["far"],
        "csi": d["csi"],
        "cost_per_1000": d["cost_per_1000_rows"],
        "value": potential_economic_value(y_true, y_prob, thr),
    }
