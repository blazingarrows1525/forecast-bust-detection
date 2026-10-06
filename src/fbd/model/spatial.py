"""S3c candidate: CNNs on the forecast window and the monsoon map, in front of S3a's head.

Everything except the image encoders is S3a's: the same 52 static inputs and
preprocessing, the same head, loss, optimiser, stopping, seeds and isotonic
calibration. So S3c - MLP measures what the images add. Frozen in
``fbd.model.params.SPATIAL_PARAMS``; registered in docs/PREREGISTRATION_S3C.md.

``branches`` picks the encoders: ("window", "map") is the primary,
("window",) and ("map",) the two registered variants. The head's input drops a
missing branch's 32 values; nothing else changes.

It runs on the GPU (``device="cuda"``), deterministically: deterministic
algorithms, cuDNN deterministic, TF32 off, a fixed cuBLAS workspace, and
weights initialised on the CPU before the move. Windows are cut on the device
from padded per-day arrays, never materialised for every row, and each map is
encoded once per distinct issue day in a batch, then gathered to its rows.
Torch is imported lazily, as in ``fbd.model.mlp``; see its note on Windows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from fbd.model import grids as G
from fbd.model.mlp import Preprocessor, _torch
from fbd.model.params import SPATIAL_PARAMS

# cuBLAS is deterministic only with a fixed workspace, set before CUDA starts.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CHUNK = 8192


def _setup(p: dict):
    torch = _torch()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(p["threads"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if p["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SPATIAL_PARAMS registers a CUDA device and none is available. "
                           "The registered result is not reproduced on another device.")
    return torch.device(p["device"])


def pool_matrix(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] adaptive average pooling along one axis, PyTorch's bins:
    start = floor(i * n_in / n_out), end = ceil((i + 1) * n_in / n_out).
    As a matrix product it is deterministic on the GPU, where the built-in
    adaptive pool's backward pass is not."""
    P = np.zeros((n_out, n_in))
    for i in range(n_out):
        a = (i * n_in) // n_out
        b = -((-(i + 1) * n_in) // n_out)
        P[i, a:b] = 1.0 / (b - a)
    return P


def map_hw_after(h: int, w: int):
    """The map encoder halves twice (padding keeps the conv sizes)."""
    return (h // 2) // 2, (w // 2) // 2


def _net(n_static: int, branches, map_hw, p: dict):
    torch = _torch()
    nn = torch.nn
    pad, k = p["conv_padding"], 3
    width = p["branch_width"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = tuple(branches)
            n_in = n_static
            if "window" in self.branches:
                self.win = nn.Sequential(
                    nn.Conv2d(G.N_WIN, 16, k, padding=pad), nn.ReLU(),
                    nn.Conv2d(16, 32, k, padding=pad), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, width, k, padding=pad), nn.ReLU())
                n_in += width
            if "map" in self.branches:
                self.map = nn.Sequential(
                    nn.Conv2d(G.N_MAP, 16, k, padding=pad), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, k, padding=pad), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 32, k, padding=pad), nn.ReLU())
                h, w = map_hw_after(*map_hw)
                self.register_buffer("pr", torch.tensor(pool_matrix(h, 4), dtype=torch.float32))
                self.register_buffer("pc", torch.tensor(pool_matrix(w, 4), dtype=torch.float32))
                self.map_fc = nn.Sequential(nn.Linear(32 * 16, width), nn.ReLU())
                n_in += width
            layers = []
            for hdim in p["hidden"]:
                layers += [nn.Linear(n_in, hdim), nn.ReLU(), nn.Dropout(p["dropout"])]
                n_in = hdim
            layers.append(nn.Linear(n_in, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, x, w=None, m=None, inv=None):
            parts = [x]
            if "window" in self.branches:
                parts.append(self.win(w).mean(dim=(2, 3)))
            if "map" in self.branches:
                z = torch.matmul(torch.matmul(self.pr, self.map(m)), self.pc.T)
                parts.append(self.map_fc(z.flatten(1))[inv])
            return self.head(torch.cat(parts, dim=1)).squeeze(1)

    return Net()


class _Feed:
    """The grids on the device, and the per-batch cutting of windows and maps."""

    def __init__(self, grids: G.Grids, wscaler, map_scaled, device):
        torch = _torch()
        self.torch = torch
        self.device = device
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt,  # noqa: E731
                                                          device=device)
        self.fc, self.era_h = t(grids.fc), t(grids.era_h)
        self.static_h, self.domain = t(grids.static_h), t(grids.domain)
        self.centre = t(grids.centre, torch.long)
        self.ar = torch.arange(G.WIN, device=device)
        self.ch3 = torch.arange(3, device=device)
        if wscaler is not None:
            self.mean = t(wscaler.mean).view(1, -1, 1, 1)
            self.std = t(wscaler.std).view(1, -1, 1, 1)
            std_mask = np.zeros(G.N_WIN, dtype=np.float32)
            std_mask[G.STANDARDISED] = 1
            self.std_mask = t(std_mask).view(1, -1, 1, 1)
        self.maps = t(map_scaled) if map_scaled is not None else None

    def windows(self, g, l_, s):
        """[B, 10, WIN, WIN] scaled windows; the same index arithmetic as
        ``Grids.windows_raw``, which the tests hold it to."""
        torch = self.torch
        rows = self.centre[s, 0].view(-1, 1, 1) + self.ar.view(1, -1, 1)
        cols = self.centre[s, 1].view(-1, 1, 1) + self.ar.view(1, 1, -1)
        dom = self.domain[rows, cols]
        ok = ((g % G.DAYS) > 0) & (l_ < G.N_LEADS - 1)
        gp = torch.where(ok, g - 1, g)
        lp = torch.where(ok, l_ + 1, l_)
        okb = ok.to(torch.float32).view(-1, 1, 1)
        this = self.fc[g.view(-1, 1, 1), l_.view(-1, 1, 1), rows, cols]
        prev = self.fc[gp.view(-1, 1, 1), lp.view(-1, 1, 1), rows, cols] * okb
        era = self.era_h[g.view(-1, 1, 1, 1), self.ch3.view(1, -1, 1, 1),
                         rows.unsqueeze(1), cols.unsqueeze(1)]
        st = self.static_h[s]
        raw = torch.cat([this.unsqueeze(1), prev.unsqueeze(1), (okb * dom).unsqueeze(1),
                         st, era], dim=1)
        present = dom.unsqueeze(1).expand_as(raw).clone()
        present[:, G.FC_PREV] = present[:, G.FC_PREV] * okb
        z = (raw - self.mean) / self.std * present
        return self.std_mask * z + (1 - self.std_mask) * raw

    def maps_for(self, g):
        """The distinct issue days' maps and, per row, which one is its."""
        days, inv = self.torch.unique(g, sorted=True, return_inverse=True)
        return self.maps[days], inv


def _logits(net, feed, branches, X, g, l_, s, idx, map_g=None):
    """Logits for rows ``idx`` without gradients, in chunks."""
    torch = _torch()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), CHUNK):
            out.append(_forward(net, feed, branches, X, g, l_, s, idx[i:i + CHUNK],
                                None if map_g is None else map_g[i:i + CHUNK]))
    return torch.cat(out)


def _forward(net, feed, branches, X, g, l_, s, b, map_g=None):
    w = feed.windows(g[b], l_[b], s[b]) if "window" in branches else None
    m = inv = None
    if "map" in branches:
        m, inv = feed.maps_for(g[b] if map_g is None else map_g)
    return net(X[b], w, m, inv)


def _train_one(feed, branches, map_hw, tr, va, seed: int, p: dict, pos_weight: float):
    torch = _torch()
    Xt, gt, lt, st, yt = tr
    Xv, gv, lv, sv, yv = va
    dev = feed.device
    torch.manual_seed(seed)
    net = _net(Xt.shape[1], branches, map_hw, p).to(dev)   # initialised on the CPU
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
    gen = torch.Generator().manual_seed(seed)
    iv = torch.arange(len(yv), device=dev)

    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, p["max_epochs"] + 1):
        net.train()
        order = torch.randperm(len(yt), generator=gen).to(dev)
        for i in range(0, len(order), p["batch_size"]):
            b = order[i:i + p["batch_size"]]
            opt.zero_grad()
            loss_fn(_forward(net, feed, branches, Xt, gt, lt, st, b), yt[b]).backward()
            opt.step()
        net.eval()
        v = float(loss_fn(_logits(net, feed, branches, Xv, gv, lv, sv, iv), yv))
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
class SpatialModel:
    branches: tuple = tuple(SPATIAL_PARAMS["branches"])
    params: dict = field(default_factory=lambda: dict(SPATIAL_PARAMS))
    grids: G.Grids | None = field(default=None, repr=False)
    features: list = field(default_factory=list)
    prep: Preprocessor | None = None
    wscaler: G.WindowScaler | None = None
    mscaler: G.MapScaler | None = None
    map_hw: tuple = (37, 47)
    nets: list = field(default_factory=list)
    calibrator: IsotonicRegression | None = None
    history: list = field(default_factory=list)

    SHUFFLE_GROUPS = ("window", "map")

    def __post_init__(self):
        self.branches = tuple(self.branches)
        if not self.branches or set(self.branches) - {"window", "map"}:
            raise ValueError(f"branches must be a non-empty subset of window, map: "
                             f"{self.branches}")
        self._feed = None

    # ---------------------------------------------------------------- inputs
    def _grids(self) -> G.Grids:
        if self.grids is None:
            self.grids = G.default_grids()
        return self.grids

    def _device_feed(self):
        if self._feed is None:
            dev = _setup(self.params)
            grids = self._grids()
            maps = self.mscaler.all_maps(grids) if self.mscaler is not None else None
            self._feed = _Feed(grids, self.wscaler, maps, dev)
        return self._feed

    def _rows(self, df: pd.DataFrame):
        torch = _torch()
        feed = self._device_feed()
        g, l_, s = self._grids().row_index(df)
        lt = lambda a: torch.as_tensor(a, dtype=torch.long, device=feed.device)  # noqa: E731
        X = torch.as_tensor(self.prep.transform(df), device=feed.device)
        return X, lt(g), lt(l_), lt(s)

    # ---------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, val: pd.DataFrame, features) -> "SpatialModel":
        torch = _torch()
        dev = _setup(self.params)
        grids = self._grids()
        self.features = list(features)
        tr = train.dropna(subset=["bust"])
        va = val.dropna(subset=["bust"])
        self.prep = Preprocessor.fit(tr, self.features)
        g, l_, s = grids.row_index(tr)
        self.wscaler = G.WindowScaler.fit(grids, g, l_, s) if "window" in self.branches else None
        self.mscaler = G.MapScaler.fit(grids, g) if "map" in self.branches else None
        self.map_hw = tuple(grids.maps.shape[-2:])
        self._feed = None
        Xt, gt, lt, st = self._rows(tr)
        Xv, gv, lv, sv = self._rows(va)
        ytr = torch.as_tensor(tr.bust.to_numpy(np.float32), device=dev)
        yva = torch.as_tensor(va.bust.to_numpy(np.float32), device=dev)
        pos_weight = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)

        self.nets, self.history = [], []
        for seed in self.params["seeds"]:
            net, info = _train_one(self._feed, self.branches, self.map_hw,
                                   (Xt, gt, lt, st, ytr), (Xv, gv, lv, sv, yva),
                                   seed, self.params, pos_weight)
            self.nets.append(net)
            self.history.append(info)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(self.predict_raw(va), va.bust.to_numpy(float))
        return self

    # ---------------------------------------------------------------- predict
    def predict_seeds(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        """[n_seeds, n]. ``shuffle="window"`` permutes windows across rows;
        ``shuffle="map"`` permutes maps across issue days (exploratory)."""
        torch = _torch()
        X, g, l_, s = self._rows(df)
        n = len(df)
        idx = torch.arange(n, device=X.device)
        map_g = None
        if shuffle == "window":
            if "window" not in self.branches:
                raise ValueError("this model has no window branch")
            perm = torch.as_tensor(np.random.default_rng(seed).permutation(n), device=X.device)
            g_w, l_w, s_w = g[perm], l_[perm], s[perm]
            map_g = g
            g, l_, s = g_w, l_w, s_w
        elif shuffle == "map":
            if "map" not in self.branches:
                raise ValueError("this model has no map branch")
            days, inv = np.unique(g.cpu().numpy(), return_inverse=True)
            perm = np.random.default_rng(seed).permutation(len(days))
            map_g = torch.as_tensor(days[perm][inv], device=X.device)
        elif shuffle is not None:
            raise ValueError(f"unknown shuffle group {shuffle!r}")
        out = []
        for net in self.nets:
            z = _logits(net, self._feed, self.branches, X, g, l_, s, idx, map_g)
            out.append(torch.sigmoid(z).cpu().numpy().astype("float64"))
        return np.vstack(out)

    def predict_raw(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        return self.predict_seeds(df, shuffle, seed).mean(axis=0)

    def predict_proba(self, df: pd.DataFrame, shuffle=None, seed=None) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df, shuffle, seed))

    # ---------------------------------------------------------------- persist
    def save(self, path) -> None:
        import joblib

        joblib.dump({"params": self.params, "branches": list(self.branches),
                     "features": self.features, "prep": self.prep, "wscaler": self.wscaler,
                     "mscaler": self.mscaler, "map_hw": list(self.map_hw),
                     "states": [{k: t.cpu() for k, t in n.state_dict().items()}
                                for n in self.nets],
                     "calibrator": self.calibrator, "history": self.history}, path)

    @classmethod
    def load(cls, path, grids: G.Grids | None = None) -> "SpatialModel":
        import joblib

        d = joblib.load(path)
        m = cls(branches=tuple(d["branches"]), params=d["params"], grids=grids,
                features=d["features"], prep=d["prep"], wscaler=d["wscaler"],
                mscaler=d["mscaler"], map_hw=tuple(d["map_hw"]),
                calibrator=d["calibrator"], history=d["history"])
        dev = _setup(m.params)
        n_static = len(m.features) + len(m.prep.flagged)
        for state in d["states"]:
            net = _net(n_static, m.branches, m.map_hw, m.params)
            net.load_state_dict(state)
            net.to(dev).eval()
            m.nets.append(net)
        return m
