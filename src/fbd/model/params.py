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
