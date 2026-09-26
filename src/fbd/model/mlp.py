"""S3a candidate: a small MLP on the incumbent's own features.

Everything that could make it differ from XGBoost other than the architecture
is held fixed: the same features, the same class weighting, the same isotonic
calibration on the validation year. Frozen in ``fbd.model.params.MLP_PARAMS``
and registered in docs/PREREGISTRATION_S3.md. Torch is imported inside the
functions that need it, so importing this module needs only numpy, pandas and
scikit-learn.

On Windows, a process must import torch *before* scikit-learn: sklearn bundles
an older msvcp140.dll, and once it is loaded torch's c10.dll cannot initialise
(WinError 1114). Entry points that train this model import torch first;
``_torch()`` turns the failure into a message that says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fbd.model.params import MLP_PARAMS


def _torch():
    try:
        import torch
    except OSError as exc:  # Windows DLL load-order failure, see the module docstring
        raise RuntimeError(
            "torch failed to load, most likely because scikit-learn was imported "
            "first on Windows. Import torch at the very top of the entry point.") from exc
    return torch


@dataclass
class Preprocessor:
    """Median imputation + missing indicators + standardisation, fitted on training rows."""
    features: list
    medians: dict
    flagged: list
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def _stack(df: pd.DataFrame, features, medians, flagged) -> np.ndarray:
        X = df[features].astype("float64")
        filled = X.fillna(medians).to_numpy(dtype="float64")
        flags = X[flagged].isna().to_numpy(dtype="float64")
        return np.hstack([filled, flags])

    @classmethod
    def fit(cls, train: pd.DataFrame, features) -> "Preprocessor":
        X = train[list(features)].astype("float64")
        medians = {c: (0.0 if pd.isna(v) else float(v)) for c, v in X.median().items()}
        flagged = [c for c in features if X[c].isna().any()]
        Z = cls._stack(train, list(features), medians, flagged)
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std == 0] = 1.0
        return cls(list(features), medians, flagged, mean, std)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        Z = self._stack(df, self.features, self.medians, self.flagged)
        return ((Z - self.mean) / self.std).astype("float32")


def _net(n_in: int, p: dict):
    nn = _torch().nn

    layers, width = [], n_in
    for h in p["hidden"]:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p["dropout"])]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _train_one(Xtr, ytr, Xva, yva, seed: int, p: dict, pos_weight: float):
    torch = _torch()

    torch.manual_seed(seed)
    net = _net(Xtr.shape[1], p)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, p["max_epochs"] + 1):
        net.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(order), p["batch_size"]):
            idx = order[i:i + p["batch_size"]]
            opt.zero_grad()
            loss_fn(net(xt[idx]).squeeze(1), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(xv).squeeze(1), yv))
        if v < best:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= p["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"seed": seed, "best_epoch": best_epoch, "val_loss": best}


@dataclass
class MLPModel:
    params: dict = field(default_factory=lambda: dict(MLP_PARAMS))
    features: list = field(default_factory=list)
    prep: Preprocessor | None = None
    nets: list = field(default_factory=list)
    calibrator: IsotonicRegression | None = None
    history: list = field(default_factory=list)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features) -> "MLPModel":
        torch = _torch()

        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.params["threads"])
        self.features = list(features)
        tr = train.dropna(subset=["bust"])
        va = val.dropna(subset=["bust"])
        self.prep = Preprocessor.fit(tr, self.features)
        Xtr, Xva = self.prep.transform(tr), self.prep.transform(va)
        ytr = tr.bust.to_numpy(np.float32)
        yva = va.bust.to_numpy(np.float32)
        pos_weight = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)

        self.nets, self.history = [], []
        for seed in self.params["seeds"]:
            net, info = _train_one(Xtr, ytr, Xva, yva, seed, self.params, pos_weight)
            self.nets.append(net)
            self.history.append(info)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(self.predict_raw(va), yva.astype(float))
        return self

    def predict_seeds(self, df: pd.DataFrame) -> np.ndarray:
        torch = _torch()

        x = torch.from_numpy(self.prep.transform(df))
        out = []
        with torch.no_grad():
            for net in self.nets:
                out.append(torch.sigmoid(net(x).squeeze(1)).numpy().astype("float64"))
        return np.vstack(out)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_seeds(df).mean(axis=0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df))

    def save(self, path) -> None:
        import joblib

        joblib.dump({"params": self.params, "features": self.features, "prep": self.prep,
                     "states": [n.state_dict() for n in self.nets],
                     "calibrator": self.calibrator, "history": self.history}, path)

    @classmethod
    def load(cls, path) -> "MLPModel":
        import joblib

        d = joblib.load(path)
        m = cls(params=d["params"], features=d["features"], prep=d["prep"],
                calibrator=d["calibrator"], history=d["history"])
        n_in = len(m.features) + len(m.prep.flagged)
        for state in d["states"]:
            net = _net(n_in, m.params)
            net.load_state_dict(state)
            net.eval()
            m.nets.append(net)
        return m
