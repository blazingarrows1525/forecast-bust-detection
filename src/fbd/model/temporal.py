"""S3b candidate: a GRU over 14 days of history in front of S3a's MLP head.

Everything except the history encoder is S3a's: the same 52 static inputs and
preprocessing, the same head, loss, optimiser, stopping, seeds and isotonic
calibration. So S3b - MLP measures what the history adds. Frozen in
``fbd.model.params.TEMPORAL_PARAMS``; registered in docs/PREREGISTRATION_S3B.md.
Torch is imported lazily, as in ``fbd.model.mlp``; see its note on Windows.

The model reads each row's history from the fold dataset it is built with
(``dataset=``), because the rows it scores are only the Day 3-7 comparison rows
while the history needs every lead. ``history`` stays what it is on the MLP:
the per-seed training log that promote.py records.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fbd.model import sequence as Q
from fbd.model.mlp import Preprocessor, _torch
from fbd.model.params import TEMPORAL_PARAMS


def _net(n_static: int, p: dict):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(Q.N_CHANNELS, p["gru_hidden"], num_layers=p["gru_layers"],
                              batch_first=True)
            layers, width = [], n_static + p["gru_hidden"]
            for h in p["hidden"]:
                layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p["dropout"])]
                width = h
            layers.append(nn.Linear(width, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, x, s):
            _out, h = self.gru(s)
            return self.head(torch.cat([x, h[-1]], dim=1)).squeeze(1)

    return Net()


def _logits(net, x, u, idx, chunk: int = 8192):
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            out.append(net(x[i:i + chunk], u[idx[i:i + chunk]]))
    return torch.cat(out)


def _train_one(tr, va, seed: int, p: dict, pos_weight: float):
    torch = _torch()
    xt, ut, it, yt = tr
    xv, uv, iv, yv = va
    torch.manual_seed(seed)
    net = _net(xt.shape[1], p)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    gen = torch.Generator().manual_seed(seed)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, p["max_epochs"] + 1):
        net.train()
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(order), p["batch_size"]):
            b = order[i:i + p["batch_size"]]
            opt.zero_grad()
            loss_fn(net(xt[b], ut[it[b]]), yt[b]).backward()
            opt.step()
        net.eval()
        v = float(loss_fn(_logits(net, xv, uv, iv), yv))
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
class TemporalModel:
    dataset: pd.DataFrame | None = field(default=None, repr=False)
    params: dict = field(default_factory=lambda: dict(TEMPORAL_PARAMS))
    features: list = field(default_factory=list)
    prep: Preprocessor | None = None
    scaler: Q.SeqScaler | None = None
    nets: list = field(default_factory=list)
    calibrator: IsotonicRegression | None = None
    history: list = field(default_factory=list)

    needs_dataset = True
    SHUFFLE_GROUPS = ("weather", "verification")

    def __post_init__(self):
        self._hist = Q.History.from_frame(self.dataset) if self.dataset is not None else None

    def _tensors(self, df: pd.DataFrame, shuffle=None, seed=None):
        torch = _torch()
        keys, idx = Q.unique_keys(df)
        u = self.scaler.transform(self._hist.raw(keys.subdivision_id, keys.init_date))
        if shuffle is not None:
            cols = Q.GROUP_CHANNELS[shuffle]
            perm = np.random.default_rng(seed).permutation(len(u))
            u[:, :, cols] = u[perm][:, :, cols]
        return (torch.from_numpy(self.prep.transform(df)), torch.from_numpy(u),
                torch.from_numpy(idx))

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features) -> "TemporalModel":
        torch = _torch()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.params["threads"])
        self.features = list(features)
        tr = train.dropna(subset=["bust"])
        va = val.dropna(subset=["bust"])
        self.prep = Preprocessor.fit(tr, self.features)
        keys, _ = Q.unique_keys(tr)
        self.scaler = Q.SeqScaler.fit(self._hist.raw(keys.subdivision_id, keys.init_date))
        xt, ut, it = self._tensors(tr)
        xv, uv, iv = self._tensors(va)
        ytr = torch.from_numpy(tr.bust.to_numpy(np.float32))
        yva = torch.from_numpy(va.bust.to_numpy(np.float32))
        pos_weight = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)

        self.nets, self.history = [], []
        for seed in self.params["seeds"]:
            net, info = _train_one((xt, ut, it, ytr), (xv, uv, iv, yva), seed, self.params,
                                   pos_weight)
            self.nets.append(net)
            self.history.append(info)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(self.predict_raw(va), yva.numpy().astype(float))
        return self

    def predict_seeds(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        torch = _torch()
        x, u, idx = self._tensors(df, shuffle, seed)
        return np.vstack([torch.sigmoid(_logits(net, x, u, idx)).numpy().astype("float64")
                          for net in self.nets])

    def predict_raw(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        return self.predict_seeds(df, shuffle, seed).mean(axis=0)

    def predict_proba(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df, shuffle, seed))

    def save(self, path) -> None:
        import joblib

        joblib.dump({"params": self.params, "features": self.features, "prep": self.prep,
                     "scaler": self.scaler, "states": [n.state_dict() for n in self.nets],
                     "calibrator": self.calibrator, "history": self.history}, path)

    @classmethod
    def load(cls, path, dataset: pd.DataFrame) -> "TemporalModel":
        import joblib

        d = joblib.load(path)
        m = cls(dataset=dataset, params=d["params"], features=d["features"], prep=d["prep"],
                scaler=d["scaler"], calibrator=d["calibrator"], history=d["history"])
        n_static = len(m.features) + len(m.prep.flagged)
        for state in d["states"]:
            net = _net(n_static, m.params)
            net.load_state_dict(state)
            net.eval()
            m.nets.append(net)
        return m
