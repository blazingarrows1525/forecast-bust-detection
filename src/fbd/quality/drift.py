"""Serving-side drift check: a fast KS test against a saved training snapshot.

**This is not a duplicate of ``fbd.mlops.drift``.** The two answer different
questions on different cadences, and keeping them separate is deliberate:

| | ``quality.drift`` (here) | ``mlops.drift`` |
|---|---|---|
| When | every ``/api/health`` call | scheduled / on demand |
| Cost | microseconds, reads a small .npz | reads the full dataset, scores it |
| Needs labels | no | yes, for the calibration check |
| Answers | "does today's input look like training?" | "should we retrain?" |

A health endpoint must be cheap and must never block, so it gets the
two-sample KS statistic against a precomputed reference sample rather than a
full PSI-plus-calibration pass.

KS rather than PSI here on purpose: KS needs no binning agreement between the
two samples, so a reference snapshot of a few thousand values is enough and no
bin edges have to be persisted alongside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fbd import config

REFERENCE_PATH = config.ARTIFACTS / "reference_distributions.npz"

# Features worth watching on the serving path. Not all 52 -- these are the ones
# that carry the most SHAP weight (D-013) plus the moisture and shear indices
# that D-016 showed actually move between years.
WATCHED_FEATURES = (
    "fcst_rain_mm",
    "fcst_anomaly",
    "lagged_spread",
    "tcwv",
    "india_tcwv_z",
    "india_shear_z",
    "clim_bust_rate",
)

# Sub-sample size per feature. Keeps the artifact small (tens of KB) while
# leaving KS plenty of power at the alpha below.
MAX_REFERENCE_SAMPLES = 5000

# 0.01 rather than 0.05: with thousands of samples KS will call statistically
# significant differences that are operationally meaningless, so the bar is
# raised to keep the health endpoint from crying wolf every morning.
DEFAULT_ALPHA = 0.01


@dataclass
class DriftStatus:
    status: str                       # OK | WATCH | DRIFT | UNKNOWN
    n_drifting: int = 0
    n_checked: int = 0
    detail: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "n_drifting": self.n_drifting,
            "n_checked": self.n_checked,
            "drifting_features": sorted(self.detail, key=self.detail.get, reverse=True)[:5],
            "note": self.note,
        }


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and p-value.

    Implemented directly rather than via scipy so the serving image does not
    need scipy at all -- requirements-serve.txt stays minimal, which is what
    keeps the offline container small.
    """
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")

    everything = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, everything, side="right") / n1
    cdf_b = np.searchsorted(b, everything, side="right") / n2
    stat = float(np.max(np.abs(cdf_a - cdf_b)))

    # Kolmogorov asymptotic survival function, truncated series.
    en = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * stat
    terms = np.arange(1, 101)
    pvalue = 2.0 * np.sum(((-1.0) ** (terms - 1)) * np.exp(-2.0 * (terms**2) * lam**2))
    return stat, float(min(max(pvalue, 0.0), 1.0))


def save_reference(frame, path: Path | None = None, seed: int = config.RANDOM_SEED) -> Path:
    """Snapshot training-year distributions for the watched features.

    Called at the end of training, so the reference always matches the model
    that shipped with it. A reference built from a different vintage than the
    deployed model would silently compare against the wrong baseline.
    """
    rng = np.random.default_rng(seed)
    payload: dict[str, np.ndarray] = {}
    for feature in WATCHED_FEATURES:
        if feature not in frame.columns:
            continue
        values = frame[feature].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size > MAX_REFERENCE_SAMPLES:
            values = rng.choice(values, MAX_REFERENCE_SAMPLES, replace=False)
        payload[feature] = values

    out = path or REFERENCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    return out


def drift_score(reference: np.ndarray, current: np.ndarray, alpha: float = DEFAULT_ALPHA) -> dict:
    stat, pvalue = _ks_2samp(np.asarray(reference, float), np.asarray(current, float))
    return {
        "ks_stat": stat,
        "pvalue": pvalue,
        "drift_detected": bool(np.isfinite(pvalue) and pvalue < alpha),
    }


def assess_batch(frame, path: Path | None = None, alpha: float = DEFAULT_ALPHA) -> DriftStatus:
    """Compare a current batch against the saved training reference."""
    ref_path = path or REFERENCE_PATH
    if not ref_path.exists():
        return DriftStatus(
            status="UNKNOWN",
            note=(
                "no reference distribution saved; re-run scripts/train_model.py "
                "to write data/artifacts/reference_distributions.npz"
            ),
        )

    detail: dict[str, float] = {}
    checked = 0
    with np.load(ref_path) as data:
        for feature in data.files:
            if feature not in getattr(frame, "columns", []):
                continue
            current = frame[feature].to_numpy(dtype=float)
            result = drift_score(data[feature], current, alpha)
            if not np.isfinite(result["ks_stat"]):
                continue
            checked += 1
            if result["drift_detected"]:
                detail[feature] = result["ks_stat"]

    if checked == 0:
        return DriftStatus(status="UNKNOWN", note="no watched feature present in the batch")

    n = len(detail)
    if n == 0:
        status, note = "OK", "input distributions consistent with training"
    elif n <= 2:
        status, note = "WATCH", f"{n} of {checked} watched features drifting"
    else:
        status, note = "DRIFT", (
            f"{n} of {checked} watched features drifting; treat probabilities "
            "as less reliable and consider retraining"
        )
    return DriftStatus(status=status, n_drifting=n, n_checked=checked, detail=detail, note=note)
