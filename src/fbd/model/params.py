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
