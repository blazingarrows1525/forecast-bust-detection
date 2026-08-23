"""Drift monitoring: does the deployed model still deserve to be trusted?

README.md already claims that calibration drift between 2021 and 2022 is "the
calibration drift the monitoring layer is designed to catch".  Until now that
monitoring layer did not exist, which made the sentence aspirational.  This
module is that layer, and running it on the project's own split reproduces the
drift the README admits to.

Four questions, in the order an operator should ask them:

1. **Has the input distribution moved?**  Population Stability Index per
   feature.  Inputs drifting is the earliest warning and needs no labels.
2. **Have the predictions moved?**  The model can emit a different probability
   distribution even on stable inputs.
3. **Is it still calibrated?**  Needs labels, so it lags by the verification
   period -- but it is the question that actually matters operationally.
4. **Is it refusing more often?**  A rising OOD rate means the atmosphere is
   moving away from what the model was fitted on.

The verdict is deliberately three-valued.  A binary pass/fail on a rare-event
model would fire constantly on sampling noise; WATCH exists so a duty roster
can note something without triggering a retrain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Verdict(str, Enum):
    OK = "OK"
    WATCH = "WATCH"
    RETRAIN = "RETRAIN"


# PSI convention from credit-risk practice, where it originates:
#   < 0.10  no material shift
#   0.10-0.25  moderate shift, worth watching
#   > 0.25  major shift
PSI_WATCH = 0.10
PSI_RETRAIN = 0.25

# Calibration: ratio of observed to predicted event rate. A model predicting
# twice the events that occur is badly miscalibrated even if it still ranks
# them correctly, because the decision cost is computed from the probability.
CAL_WATCH = 0.25   # 25% relative error
CAL_RETRAIN = 0.50

# OOD refusal rate. The detector is fitted at the 99.5th percentile, so ~0.5-1%
# is by construction. Several times that means the regime has moved.
OOD_WATCH = 0.03
OOD_RETRAIN = 0.08


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between a reference and a current sample.

    Quantile bins are taken from the *reference*, which is the whole point: we
    are asking how the new data sits inside the old distribution's shape.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < bins or cur.size < bins:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0  # a near-constant feature cannot meaningfully drift
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(ref, bins=edges)[0] / ref.size
    cur_frac = np.histogram(cur, bins=edges)[0] / cur.size

    # Floor empty bins: PSI is undefined when a bin empties, and an unfloored
    # zero makes the statistic explode to infinity on one sparse bin.
    eps = 1e-6
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


@dataclass
class DriftReport:
    """Everything an operator needs to decide whether to act."""

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    feature_psi: dict[str, float] = field(default_factory=dict)
    prediction_psi: float | None = None
    calibration_ratio: float | None = None
    observed_rate: float | None = None
    predicted_rate: float | None = None
    ood_rate: float | None = None
    n_reference: int = 0
    n_current: int = 0

    def top_drifting(self, k: int = 8) -> list[tuple[str, float]]:
        ranked = sorted(
            ((f, v) for f, v in self.feature_psi.items() if np.isfinite(v)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:k]

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "prediction_psi": self.prediction_psi,
            "calibration_ratio": self.calibration_ratio,
            "observed_rate": self.observed_rate,
            "predicted_rate": self.predicted_rate,
            "ood_rate": self.ood_rate,
            "top_drifting_features": [
                {"feature": f, "psi": round(v, 4)} for f, v in self.top_drifting()
            ],
        }


def _escalate(current: Verdict, proposed: Verdict) -> Verdict:
    order = {Verdict.OK: 0, Verdict.WATCH: 1, Verdict.RETRAIN: 2}
    return proposed if order[proposed] > order[current] else current


def assess(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    proba_col: str | None = None,
    label_col: str | None = None,
    ood_col: str | None = None,
) -> DriftReport:
    """Compare a current window against the training reference.

    Every input is optional except the features, because in live operation
    labels arrive days later than the forecasts they verify -- the monitor has
    to say something useful before then.
    """
    report = DriftReport(
        verdict=Verdict.OK, n_reference=len(reference), n_current=len(current)
    )

    # 1. Feature drift
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi = population_stability_index(
            reference[feature].to_numpy(), current[feature].to_numpy()
        )
        report.feature_psi[feature] = psi

    severe = [f for f, v in report.feature_psi.items() if np.isfinite(v) and v > PSI_RETRAIN]
    moderate = [
        f for f, v in report.feature_psi.items()
        if np.isfinite(v) and PSI_WATCH < v <= PSI_RETRAIN
    ]
    if severe:
        report.verdict = _escalate(report.verdict, Verdict.RETRAIN)
        report.reasons.append(
            f"{len(severe)} feature(s) with PSI > {PSI_RETRAIN}: {', '.join(severe[:5])}"
        )
    elif moderate:
        report.verdict = _escalate(report.verdict, Verdict.WATCH)
        report.reasons.append(
            f"{len(moderate)} feature(s) with moderate PSI: {', '.join(moderate[:5])}"
        )

    # 2. Prediction drift
    if proba_col and proba_col in reference.columns and proba_col in current.columns:
        report.prediction_psi = population_stability_index(
            reference[proba_col].to_numpy(), current[proba_col].to_numpy()
        )
        if np.isfinite(report.prediction_psi) and report.prediction_psi > PSI_RETRAIN:
            report.verdict = _escalate(report.verdict, Verdict.RETRAIN)
            report.reasons.append(
                f"predicted-probability distribution shifted (PSI {report.prediction_psi:.3f})"
            )

    # 3. Calibration drift -- the one that changes decisions
    if (
        label_col and proba_col
        and label_col in current.columns and proba_col in current.columns
    ):
        subset = current[[label_col, proba_col]].dropna()
        if len(subset) >= 100:
            observed = float(subset[label_col].mean())
            predicted = float(subset[proba_col].mean())
            report.observed_rate, report.predicted_rate = observed, predicted
            if predicted > 0:
                ratio = observed / predicted
                report.calibration_ratio = ratio
                rel = abs(ratio - 1.0)
                direction = "over-predicting" if ratio < 1 else "under-predicting"
                if rel > CAL_RETRAIN:
                    report.verdict = _escalate(report.verdict, Verdict.RETRAIN)
                    report.reasons.append(
                        f"calibration off by {rel:.0%} ({direction}): "
                        f"observed {observed:.3%} vs predicted {predicted:.3%}"
                    )
                elif rel > CAL_WATCH:
                    report.verdict = _escalate(report.verdict, Verdict.WATCH)
                    report.reasons.append(
                        f"calibration drifting {rel:.0%} ({direction}): "
                        f"observed {observed:.3%} vs predicted {predicted:.3%}"
                    )

    # 4. Refusal rate
    if ood_col and ood_col in current.columns:
        rate = float(current[ood_col].mean())
        report.ood_rate = rate
        if rate > OOD_RETRAIN:
            report.verdict = _escalate(report.verdict, Verdict.RETRAIN)
            report.reasons.append(f"OOD refusal rate {rate:.2%} far above design (~1%)")
        elif rate > OOD_WATCH:
            report.verdict = _escalate(report.verdict, Verdict.WATCH)
            report.reasons.append(f"OOD refusal rate {rate:.2%} elevated")

    if not report.reasons:
        report.reasons.append("no material drift detected")
    return report
