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


#: S3c candidate, frozen before any run (spec 2026-09-28, docs/PREREGISTRATION_S3C.md).
#: Every channel, scaling, layer, training setting and variant is here, so changing
#: any of them changes spatial_params_sha256. ``device`` is "cuda": the one amendment
#: to the spec, made before any spatial model was built (PREREGISTRATION_S3C.md §3).
SPATIAL_PARAMS = dict(
    window_size=13,
    window_grid="HRES 0.703 deg, 45 x 49, zero-padded by 6 cells",
    window_centre="area-weighted centre of the subdivision's HRES cells, rounded half up",
    window_channels=["log1p tp24, run t, lead L", "log1p tp24, run t-1, lead L+1",
                     "previous run present", "outline: cell area fraction inside s",
                     "orography (ERA5 z_sfc, bilinear)", "land-sea mask (bilinear)",
                     "in HRES domain", "tcwv at t 00Z (bilinear)",
                     "u850 at t 00Z (bilinear)", "v850 at t 00Z (bilinear)"],
    window_standardised=[0, 1, 4, 7, 8, 9],
    window_scaling="per standardised channel, mean and sd of the present in-domain "
                   "entries of the training rows' windows; absent -> 0",
    map_grid="ERA5 1.5 deg, 37 x 47, -9..45 N, 40.5..109.5 E, t 00Z",
    map_channels=["tcwv", "mslp", "u850", "v850", "q850", "z500", "u200",
                  "orography", "land-sea mask"],
    map_scaling="dynamic: per-cell mean over training issue days subtracted, divided by "
                "the channel sd of those anomalies; orography standardised; mask unscaled",
    conv_padding=1,
    window_encoder="conv3x3 10->16, relu, conv3x3 16->32, relu, maxpool2, "
                   "conv3x3 32->32, relu, global average pool -> 32",
    map_encoder="conv3x3 9->16, relu, maxpool2, conv3x3 16->32, relu, maxpool2, "
                "conv3x3 32->32, relu, adaptive average pool 4x4 (pooling matrices), "
                "linear 512->32, relu -> 32",
    branch_width=32,
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
    dtype="float32, TF32 off",
    device="cuda",
    determinism="torch.use_deterministic_algorithms, cudnn deterministic, "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8; weights initialised on the CPU",
    loss="bce with pos_weight = negatives / positives on training rows",
    stopping="validation-year weighted bce, restore best epoch",
    calibration="isotonic on the validation year, on the seed-averaged probability",
    branches=["window", "map"],
    variants={"window_only": ["window"], "map_only": ["map"]},
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
