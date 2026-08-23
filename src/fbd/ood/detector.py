"""Out-of-distribution detector: refuse to score states unlike anything in training.

Motivation is the strongest single argument for this project (strategy doc,
RED 3 / BLUE 3).  *Science Advances* (2026) showed AI weather models degrade most
on record-breaking extremes -- exactly the high-stakes days.  A bust model
trained on 2016-2020 has the same weakness.  The correct behaviour on an
unprecedented state is not a confident number; it is **"conditions outside
training experience, confidence unavailable."**

Method: Mahalanobis distance in the standardised feature space, fitted on the
training rows only.  Chosen over an isolation forest because it is explainable --
we can say *which* feature is anomalous and by how many standard deviations,
which the reason panel needs.

Refusing is only defensible if refused rows are genuinely harder.  `validate()`
checks exactly that: if the model's skill on refused rows is not worse than on
accepted rows, the detector is refusing at random and should be removed rather
than kept as decoration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fbd import config


class MahalanobisOOD:
    def __init__(self, quantile: float = 0.995, shrinkage: float = 1e-3):
        self.quantile = quantile
        self.shrinkage = shrinkage
        self.features_: list[str] = []
        self.mu_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None
        self.threshold_: float = np.inf
        self.sd_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, features: list[str]) -> "MahalanobisOOD":
        self.features_ = list(features)
        X = train[self.features_].to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        self.mu_ = np.nanmean(X, axis=0)
        self.sd_ = np.nanstd(X, axis=0)
        self.sd_[self.sd_ == 0] = 1.0

        Z = (np.nan_to_num(X, nan=0.0) - self.mu_) / self.sd_
        cov = np.cov(Z, rowvar=False)
        cov += self.shrinkage * np.eye(cov.shape[0])  # keep it invertible
        self.prec_ = np.linalg.pinv(cov)

        d = self._distance(train)
        self.threshold_ = float(np.nanquantile(d, self.quantile))
        return self

    def _distance(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features_].to_numpy(dtype=float)
        Z = (np.nan_to_num(X, nan=np.nan) - self.mu_) / self.sd_
        Z = np.nan_to_num(Z, nan=0.0)
        return np.sqrt(np.einsum("ij,jk,ik->i", Z, self.prec_, Z, optimize=True))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self._distance(df)

    def is_ood(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df) > self.threshold_

    def top_anomalous_features(self, row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
        """Which features make this state unusual, in standard deviations."""
        x = np.array([row.get(f, np.nan) for f in self.features_], dtype=float)
        z = (x - self.mu_) / self.sd_
        z = np.nan_to_num(z, nan=0.0)
        order = np.argsort(-np.abs(z))[:k]
        return [(self.features_[i], float(z[i])) for i in order]


def validate(
    df: pd.DataFrame, probs: np.ndarray, is_ood: np.ndarray
) -> pd.DataFrame:
    """Are refused rows actually harder?  If not, the detector is decoration."""
    from fbd.evaluate import metrics as M

    out = []
    for name, mask in (("accepted", ~is_ood), ("refused (OOD)", is_ood)):
        if mask.sum() < 20:
            out.append({"group": name, "n": int(mask.sum())})
            continue
        y = df.bust.to_numpy()[mask]
        p = probs[mask]
        out.append(
            {
                "group": name,
                "n": int(mask.sum()),
                "bust_rate": float(np.mean(y)),
                "auroc": M.auroc(y, p),
                "brier": M.brier(y, p),
                "mean_abs_error_mm": float(df.abs_error.to_numpy()[mask].mean()),
            }
        )
    return pd.DataFrame(out)
