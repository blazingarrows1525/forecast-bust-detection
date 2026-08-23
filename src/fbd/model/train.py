"""Gradient-boosted bust model + isotonic calibration.

Model family is fixed by LOGIC.md sec 6 and is not up for negotiation on
aesthetic grounds: the problem statement mandates explainable output naming the
meteorological reasons, the labelled sample is ~10^5 rows, and busts are rare.
SHAP on trees delivers the mandated per-flag reasons directly; a CNN does not.

Protocol
--------
  fit        on TRAIN_YEARS   (2016-2020)
  calibrate  on VAL_YEARS     (2021)  -- isotonic, never fitted on test
  evaluate   on TEST_YEARS    (2022)  -- touched once, at the end

Subdivision identity is deliberately **not** one-hot encoded.  Instead the model
sees the physical attributes that make a subdivision what it is (elevation,
terrain roughness, coastal fraction, latitude) plus its climatological bust rate.
That keeps the model from memorising "Konkan busts a lot" and lets it say
something about a region it has not seen, which is what a deployable system
needs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fbd import config

# Feature groups.  Any column absent from the dataset is skipped, so the model
# trains with or without the ERA5 stage having completed.
FORECAST_FEATURES = [
    "lead_day",
    "fcst_rain_mm",
    "fcst_anomaly",
    "fcst_rel_to_p90",
    "lagged_spread",
    "lagged_spread_rel",
    "lagged_mean",
    "lagged_range",
    "lagged_n_members",
    "spread_growth",
    "jumpiness",
    "fcst_prev_run",
    "clim_obs_mean",
    "clim_obs_p90",
    "clim_fcst_mean",
    "clim_bust_rate",
    "day_of_season",
    "month",
]

ERA5_LOCAL_FEATURES = [
    "moisture_flux_850", "wind_shear", "tcwv", "z500", "mslp", "u850", "v850",
]

ERA5_NATIONAL_FEATURES = [
    "somali_jet_z", "monsoon_trough_mslp_z", "nw_z500_z", "india_shear_z",
    "india_tcwv_z", "india_q850_z", "mcz_q850_z", "bob_vorticity_max_z",
    "bob_mslp_min_z",
    "somali_jet_d1", "somali_jet_d3", "bob_vorticity_max_d1",
    "monsoon_trough_mslp_d1", "nw_z500_d1", "india_tcwv_d1",
]

REGIME_FEATURES = [f"regime_{r}" for r in config.REGIMES] + [
    "regime_entropy",
    "regime_top_prob",
]

STATIC_FEATURES = [
    "elevation_m", "terrain_roughness_m", "coastal_index", "orographic_index",
]

ALL_FEATURES = (
    FORECAST_FEATURES
    + ERA5_LOCAL_FEATURES
    + ERA5_NATIONAL_FEATURES
    + REGIME_FEATURES
    + STATIC_FEATURES
)


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURES if c in df.columns]


@dataclass
class BustModel:
    features: list[str] = field(default_factory=list)
    booster: object = None
    calibrator: IsotonicRegression | None = None
    prior: float = 0.0
    params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame | None = None,
        features: list[str] | None = None,
        **overrides,
    ) -> "BustModel":
        import xgboost as xgb

        self.features = features or available_features(train)
        tr = train.dropna(subset=["bust"])
        X = tr[self.features].to_numpy(dtype=np.float32)
        y = tr.bust.to_numpy(dtype=int)
        self.prior = float(y.mean())

        pos = max(int(y.sum()), 1)
        neg = int((y == 0).sum())
        params = dict(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=20,
            reg_lambda=2.0,
            # Busts are rare; class weighting is mandated by LOGIC.md sec 4.5.
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.RANDOM_SEED,
            n_jobs=0,
        )
        params.update(overrides)
        self.params = params

        self.booster = xgb.XGBClassifier(**params)
        eval_set = None
        if val is not None and len(val):
            v = val.dropna(subset=["bust"])
            eval_set = [(v[self.features].to_numpy(np.float32), v.bust.to_numpy(int))]
        self.booster.fit(X, y, eval_set=eval_set, verbose=False)

        # Isotonic calibration on the validation year.  scale_pos_weight makes
        # the raw scores systematically over-confident, so calibration is not
        # optional here -- an uncalibrated bust probability is worse than none.
        if val is not None and len(val):
            v = val.dropna(subset=["bust"])
            raw = self.booster.predict_proba(
                v[self.features].to_numpy(np.float32)
            )[:, 1]
            self.calibrator = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            ).fit(raw, v.bust.to_numpy(float))
        return self

    # -------------------------------------------------------------- predict
    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=np.float32)
        return self.booster.predict_proba(X)[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(df)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    # ----------------------------------------------------------------- io
    def save(self, path) -> None:
        import joblib

        joblib.dump(
            {
                "features": self.features,
                "booster": self.booster,
                "calibrator": self.calibrator,
                "prior": self.prior,
                "params": self.params,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "BustModel":
        import joblib

        d = joblib.load(path)
        m = cls(
            features=d["features"],
            booster=d["booster"],
            calibrator=d["calibrator"],
            prior=d["prior"],
            params=d.get("params", {}),
        )
        return m


def split_frames(ds: pd.DataFrame):
    tr = ds[ds.split == "train"]
    va = ds[ds.split == "val"]
    te = ds[ds.split == "test"]
    return tr, va, te
