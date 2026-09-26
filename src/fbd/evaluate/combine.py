"""S1c: does the model plus ENS spread outrank ENS spread alone?

Registered in docs/PREREGISTRATION_S1C.md. The combiner is S1's secondary (b)
with one correction: it takes the model's *uncalibrated* probability, because
the isotonic calibration was itself fitted on the validation year, which would
make calibrated inputs in-sample for the combiner. The raw score of a model
trained on the training years is out of sample on the validation year.

The row selection here is pure pandas (tested in CI); scikit-learn is imported
only when a combiner is fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fbd import config
from fbd.evaluate import ens as E
from fbd.model.params import COMBINER_PARAMS

#: Registered: 2022 is reported beside the primary, never in it (S1 saw a
#: combination on 2022).
PRIMARY_YEARS = (2019, 2020, 2021)
YEARS = (2019, 2020, 2021, 2022)
TEXT = {
    "model_better": "model + ENS spread outranks ENS spread alone on average over 2019–2021",
    "ens_better": "ENS spread alone outranks the model + ENS combination on average over "
                  "2019–2021",
    "indistinguishable": "the model adds nothing detectable to ENS spread on average over "
                         "2019–2021",
}


def validation_rows(ds: pd.DataFrame, ens: pd.DataFrame,
                    lead_days=config.DECISION_BAND) -> pd.DataFrame:
    """The rows a fold's combiner is fitted on: its validation year, decision band."""
    _test, fit = E.comparison_rows(ds, ens, lead_days=lead_days)
    return fit[(fit.split == "val") & fit.lead_day.isin(list(lead_days))]


def _design(p_raw, spread) -> np.ndarray:
    clip = COMBINER_PARAMS["clip"]
    p = np.clip(np.asarray(p_raw, dtype=float), clip, 1 - clip)
    return np.column_stack([np.log(p / (1 - p)), np.log1p(np.asarray(spread, dtype=float))])


class Combiner:
    """Logistic regression on [logit p_model_raw, log1p ENS spread]."""

    def __init__(self):
        self.lr = None

    def fit(self, p_raw, spread, y) -> "Combiner":
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=COMBINER_PARAMS["C"], max_iter=COMBINER_PARAMS["max_iter"])
        self.lr.fit(_design(p_raw, spread), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, p_raw, spread) -> np.ndarray:
        return self.lr.predict_proba(_design(p_raw, spread))[:, 1]

    def coefficients(self) -> dict:
        return {"logit_p_model": float(self.lr.coef_[0][0]),
                "log1p_ens_spread": float(self.lr.coef_[0][1]),
                "intercept": float(self.lr.intercept_[0])}
