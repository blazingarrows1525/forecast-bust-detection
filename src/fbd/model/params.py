"""The model's hyperparameters, in one pure place so S1b can register them.

``scale_pos_weight`` is not here: it is computed from each training set's
class balance (LOGIC.md sec 4.5), so it differs by fold by design.
"""
from __future__ import annotations

import hashlib
import json

from fbd import config

DEFAULT_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.75,
    min_child_weight=20,
    reg_lambda=2.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=config.RANDOM_SEED,
    n_jobs=0,
)


def params_sha256(params: dict = DEFAULT_PARAMS) -> str:
    blob = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


#: S3a candidate, frozen before any run (spec §5, docs/PREREGISTRATION_S3.md).
MLP_PARAMS = dict(
    hidden=[128, 64],
    activation="relu",
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=1024,
    max_epochs=60,
    patience=5,
    seeds=[20260920, 20260921, 20260922, 20260923, 20260924],
    threads=8,
    dtype="float32",
    device="cpu",
    missing="training-rows median + indicator for features missing in training",
    scaling="training-rows mean and sd after imputation; sd 0 -> 1",
    loss="bce with pos_weight = negatives / positives on training rows",
    stopping="validation-year weighted bce, restore best epoch",
    calibration="isotonic on the validation year, on the seed-averaged probability",
)


#: S3b candidate, frozen before any run (spec 2026-09-27, docs/PREREGISTRATION_S3B.md).
#: The channel lists, window, lag and leads are here so that changing any of them
#: changes temporal_params_sha256.
TEMPORAL_PARAMS = dict(
    window=14,
    verify_lag=2,
    verify_leads=[1, 3, 5],
    weather=["moisture_flux_850", "wind_shear", "u850", "v850", "z500", "tcwv", "mslp",
             "onshore_wind", "somali_jet_z", "monsoon_trough_mslp_z", "nw_z500_z",
             "india_shear_z", "india_tcwv_z", "india_q850_z", "mcz_q850_z",
             "bob_vorticity_max_z", "bob_mslp_min_z"],
    verification=["obs_rain_mm on d - 2", "error and bust of leads 1, 3, 5 verifying on d - 2"],
    sequence_scaling="training-rows mean and sd of present entries per value channel; "
                     "absent -> 0 after scaling; 5 presence masks, unscaled",
    gru_hidden=32,
    gru_layers=1,
    static="S3a Preprocessor on the incumbent's 52 features",
    hidden=[128, 64],
    activation="relu",
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=1024,
    max_epochs=60,
    patience=5,
    seeds=[20260920, 20260921, 20260922, 20260923, 20260924],
    threads=8,
    dtype="float32",
    device="cpu",
    loss="bce with pos_weight = negatives / positives on training rows",
    stopping="validation-year weighted bce, restore best epoch",
    calibration="isotonic on the validation year, on the seed-averaged probability",
)


#: S1c combiner, frozen before any run (docs/PREREGISTRATION_S1C.md).
COMBINER_PARAMS = dict(
    inputs=["logit of the uncalibrated model probability", "log(1 + ENS spread)"],
    model="logistic regression, lbfgs",
    C=1e6,
    max_iter=1000,
    clip=1e-6,
    fit_rows="the fold's validation year, Day 3-7, rows with a label and ENS spread",
)
