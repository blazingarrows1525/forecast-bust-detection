"""B3 and B4: two registered refinements of the served combiner (D-029, D-030).

docs/PREREGISTRATION_B3.md and docs/PREREGISTRATION_B4.md. The rules are pure;
scikit-learn is imported only when a combiner is fitted.
"""
from __future__ import annotations

import numpy as np

from fbd.model.params import COMBINER_PARAMS

YEARS = (2019, 2020, 2021, 2022)
SHORT_LEADS = (1, 2)
TOP_SHARE = 0.10


def tail_gap(y_true, prob, share: float = TOP_SHARE) -> float:
    """|mean predicted - observed rate| over the top ``share`` of rows by prediction."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    k = max(1, int(np.ceil(share * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    return float(abs(p[top].mean() - y[top].mean()))


def b4_verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "short_lead_fit_better"
    if hi < 0:
        return "extrapolation_better"
    return "indistinguishable"


B4_TEXT = {
    "short_lead_fit_better": "a combiner fitted on Days 1–2 outranks the served extrapolation "
                             "at Days 1–2 across 2019–2022",
    "extrapolation_better": "the served (Day 3–7-fitted) combiner outranks a Day 1–2 fit at "
                            "Days 1–2 across 2019–2022",
    "indistinguishable": "a Day 1–2 combiner is not distinguishable from the served "
                         "extrapolation at Days 1–2 across 2019–2022",
}


def b3_verdict(tail: dict, brier: dict) -> str:
    """Tail gap and Brier margins are candidate - incumbent; lower is better for both."""
    if tail["lo"] > 0 or brier["lo"] > 0:
        return "reject"
    if tail["hi"] < 0 and brier["lo"] <= 0:
        return "adopt"
    return "not_established"


B3_TEXT = {
    "adopt": "the beta-family combiner narrows the top-decile reliability gap without a "
             "worse Brier score across 2019–2022",
    "reject": "the beta-family combiner widens the tail gap or worsens the Brier score "
              "across 2019–2022",
    "not_established": "a tail-calibration gain from the beta-family combiner is not "
                       "established across 2019–2022",
}


def _beta_design(p_raw, spread) -> np.ndarray:
    clip = COMBINER_PARAMS["clip"]
    p = np.clip(np.asarray(p_raw, dtype=float), clip, 1 - clip)
    return np.column_stack([np.log(p), np.log1p(-p), np.log1p(np.asarray(spread, dtype=float))])


class BetaCombiner:
    """Logistic regression on [ln p_raw, ln(1 - p_raw), log1p ENS spread].

    Nests the served form: coefficients a and -a on the first two inputs give
    a * logit(p_raw), so it can bend the tail without being made to."""

    def __init__(self):
        self.lr = None

    def fit(self, p_raw, spread, y) -> "BetaCombiner":
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=COMBINER_PARAMS["C"], max_iter=COMBINER_PARAMS["max_iter"])
        self.lr.fit(_beta_design(p_raw, spread), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, p_raw, spread) -> np.ndarray:
        return self.lr.predict_proba(_beta_design(p_raw, spread))[:, 1]

    def coefficients(self) -> dict:
        c = self.lr.coef_[0]
        return {"ln_p": float(c[0]), "ln_1mp": float(c[1]), "log1p_ens_spread": float(c[2]),
                "intercept": float(self.lr.intercept_[0])}
