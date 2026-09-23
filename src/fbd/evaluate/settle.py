"""The statistics registered in docs/PREREGISTRATION_S1.md. Needs scikit-learn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fbd.evaluate import metrics as M
from fbd.evaluate import registration as R
from fbd.evaluate import uncertainty as U
from fbd.model import baselines as B
from fbd.quality.escalation import REVIEW_THRESHOLD


def choose_comparator(y, raw, rel):
    """The published rule: face whichever ENS variant ranks busts better."""
    return ("raw", raw) if M.auroc(y, raw) >= M.auroc(y, rel) else ("relative", rel)


def interval(iv) -> dict:
    return {**iv.as_dict(), "excludes_zero": iv.excludes_zero}


def primary(y, p_model, comparator, clusters, n_boot, seed) -> dict:
    iv = U.paired_difference(M.auroc, y, p_model, comparator, clusters,
                             n_boot=n_boot, seed=seed)
    v = R.verdict(iv.lo, iv.hi)
    return {**interval(iv), "verdict": v, "text": R.VERDICT_TEXT[v],
            "n_rows": int(len(y)), "n_init_dates": int(len(np.unique(clusters))),
            "model_auroc": M.auroc(y, p_model), "ens_auroc": M.auroc(y, comparator)}


def _cost(y, p):
    return M.decision_cost(y, p, REVIEW_THRESHOLD)["cost_per_1000_rows"]


def secondary_calibrated(test, fit, p_model, clusters, n_boot, seed) -> dict:
    """(a) ENS isotonic fitted on 2021 only, the model's calibration year (D-010)."""
    fit21 = fit[pd.to_datetime(fit.init_date).dt.year == 2021]
    p_ens = B.SpreadBaseline("ens_spread").fit(fit21).predict_proba(test)
    y = test.bust.to_numpy(float)
    out = {"fit_rows_2021": int(len(fit21))}
    for name, fn in (("auroc", M.auroc), ("brier", M.brier),
                     ("bss", M.brier_skill_score), ("cost_per_1000", _cost)):
        out[name] = interval(U.paired_difference(fn, y, p_model, p_ens, clusters,
                                                 n_boot=n_boot, seed=seed))
    out["note"] = ("model minus ENS calibrated on 2021; lower Brier and cost are "
                   f"better; cost at the review threshold {REVIEW_THRESHOLD:.4f}")
    return out


def _design(p, spread):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1 - p)), np.log1p(np.asarray(spread, float))])


def secondary_combined(test, fit21, p_model, p_model_21, clusters, n_boot, seed) -> dict:
    """(b) Does the model add anything on top of ENS? Fitted on 2021, tested on 2022."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_design(p_model_21, fit21.ens_spread), fit21.bust.to_numpy(int))
    p_combo = lr.predict_proba(_design(p_model, test.ens_spread))[:, 1]
    y = test.bust.to_numpy(float)
    raw = test.ens_spread.to_numpy(float)
    return {
        "fit_rows_2021": int(len(fit21)),
        "coefficients": {"logit_p_model": float(lr.coef_[0][0]),
                         "log1p_ens_spread": float(lr.coef_[0][1]),
                         "intercept": float(lr.intercept_[0])},
        "combined_minus_ens": interval(U.paired_difference(
            M.auroc, y, p_combo, raw, clusters, n_boot=n_boot, seed=seed)),
        "combined_minus_model": interval(U.paired_difference(
            M.auroc, y, p_combo, p_model, clusters, n_boot=n_boot, seed=seed)),
        "caveat": ("The model's isotonic calibration was fitted on 2021, so this "
                   "combination is fitted on probabilities in-sample for calibration. "
                   "Evaluation on 2022 is out of sample."),
    }


def by_group(y, p_model, comparator, groups, clusters, n_boot, seed) -> dict:
    """Exploratory: the margin within each group (lead, month, old/new dates)."""
    out = {}
    groups = np.asarray(groups)
    for g in sorted(np.unique(groups), key=str):
        m = groups == g
        if len(np.unique(y[m])) < 2:
            continue
        out[str(g)] = interval(U.paired_difference(
            M.auroc, y[m], p_model[m], comparator[m], clusters[m],
            n_boot=n_boot, seed=seed))
    return out
