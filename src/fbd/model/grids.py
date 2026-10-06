"""S3c: the gridded inputs a spatial candidate reads (spec 2026-09-28 §3). numpy/xarray only.

Two images per row (subdivision s, issue day t, lead L):

- **the window**: 13 x 13 cells of the HRES grid centred on s, 10 channels;
- **the map**: the ERA5 monsoon-region map at t 00Z, 9 channels.

Nothing issued or analysed after t 00Z is read: the runs are those issued on t
and t - 1, and the analysis is t 00Z, the same "state at issue time" rule the
52 features follow (LOGIC.md 7.1). Causality is index arithmetic on dense
per-day arrays, and the tests and the audit check it by poisoning.

Days are indexed globally: g = (year - first year) * 122 + days since 1 June.
Frozen in ``fbd.model.params.SPATIAL_PARAMS``; registered in
docs/PREREGISTRATION_S3C.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fbd import config
from fbd.evaluate import provenance as P
from fbd.model.params import SPATIAL_PARAMS

WIN = int(SPATIAL_PARAMS["window_size"])
PAD = WIN // 2
DAYS = 122                       # 1 June .. 30 September
N_LEADS = 10
N_WIN = len(SPATIAL_PARAMS["window_channels"])
N_MAP = len(SPATIAL_PARAMS["map_channels"])
N_MAP_DYN = N_MAP - 2            # the last two map channels are static
STANDARDISED = list(SPATIAL_PARAMS["window_standardised"])
#: window channel indices
FC, FC_PREV, PREV_OK, OUTLINE, OROG, LSM, IN_DOMAIN, TCWV, U850, V850 = range(N_WIN)
#: a cell's area on the 0.703 deg HRES grid needs the earth's radius (authalic, km)
EARTH_KM = 6371.0072
STEP_DEG = 0.703125

HRES_DIR = config.WB2_RAW / "hres"
ERA5_DIR = config.WB2_RAW / "era5"
WEIGHTS = config.INTERIM / "weights_45x49_6.6797_37.6172_66.0938_99.8438.parquet"
MANIFEST = config.ARTIFACTS / "s3c_inputs.json"


# ------------------------------------------------------------------ provenance
def input_files(years=config.ALL_YEARS) -> list:
    """The 23 files S3c reads: HRES x 7, ERA5 2-D x 7, ERA5 3-D x 7, static, weights."""
    out = [HRES_DIR / f"hres_tp24_india_{y}.nc" for y in years]
    out += [ERA5_DIR / f"era5_2d_{y}.nc" for y in years]
    out += [ERA5_DIR / f"era5_3d_{y}.nc" for y in years]
    return out + [ERA5_DIR / "era5_static.nc", WEIGHTS]


def write_manifest(path: Path = MANIFEST, files=None, root: Path = config.ROOT) -> dict:
    files = input_files() if files is None else files
    root = Path(root).resolve()
    body = {Path(f).resolve().relative_to(root).as_posix(): P.sha256_file(f) for f in files}
    Path(path).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def verify_manifest(path: Path = MANIFEST, root: Path = config.ROOT) -> list:
    """Every listed file still hashes to its pin. Returns the problems, empty if none."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for rel, want in sorted(body.items()):
        f = Path(root) / rel
        got = P.sha256_file(f) if f.exists() else "missing"
        if got != want:
            problems.append(f"{rel}: pinned {want}, on disk {got}")
    return problems


# ------------------------------------------------------------------ geometry
def bilinear_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """[len(dst), len(src)] weights of linear interpolation along one ascending axis.

    Bilinear interpolation on a regular grid is separable: F_dst = A_lat F A_lon^T.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if np.any(np.diff(src) <= 0):
        raise ValueError("source axis must be strictly ascending")
    if dst.min() < src[0] or dst.max() > src[-1]:
        raise ValueError("target outside the source axis; no extrapolation")
    A = np.zeros((len(dst), len(src)))
    hi = np.clip(np.searchsorted(src, dst, side="right"), 1, len(src) - 1)
    lo = hi - 1
    w = (dst - src[lo]) / (src[hi] - src[lo])
    A[np.arange(len(dst)), lo] = 1.0 - w
    A[np.arange(len(dst)), hi] += w
    return A


def cell_area_km2(lat: np.ndarray, step: float = STEP_DEG) -> np.ndarray:
    """Area of a step x step cell centred on each latitude, on a sphere."""
    lat = np.radians(np.asarray(lat, float))
    h = np.radians(step) / 2
    return EARTH_KM ** 2 * np.radians(step) * (np.sin(lat + h) - np.sin(lat - h))


def centres(weights: pd.DataFrame, subs: list) -> np.ndarray:
    """[S, 2] (row, col): each subdivision's area-weighted centre, rounded half up."""
    out = np.zeros((len(subs), 2), dtype=np.int64)
    for i, s in enumerate(subs):
        w = weights[weights.subdivision_id == s]
        a = w.weight_km2.to_numpy(float)
        out[i, 0] = int(np.floor((w.lat_idx.to_numpy(float) * a).sum() / a.sum() + 0.5))
        out[i, 1] = int(np.floor((w.lon_idx.to_numpy(float) * a).sum() / a.sum() + 0.5))
    return out


def outlines(weights: pd.DataFrame, subs: list, lat: np.ndarray, shape) -> np.ndarray:
    """[S, H, W]: the fraction of each HRES cell's area inside each subdivision."""
    area = cell_area_km2(lat)
    out = np.zeros((len(subs),) + tuple(shape))
    index = {s: i for i, s in enumerate(subs)}
    w = weights[weights.subdivision_id.isin(index)]
    si = w.subdivision_id.map(index).to_numpy()
    r, c = w.lat_idx.to_numpy(), w.lon_idx.to_numpy()
    np.add.at(out, (si, r, c), w.weight_km2.to_numpy(float) / area[r])
    return np.clip(out, 0.0, 1.0)


def pad(a: np.ndarray) -> np.ndarray:
    """Zero-pad the last two axes by PAD cells."""
    widths = [(0, 0)] * (a.ndim - 2) + [(PAD, PAD), (PAD, PAD)]
    return np.pad(a, widths)


# ------------------------------------------------------------------ the store
@dataclass
class Grids:
    """Dense per-day arrays for every season, padded for window cutting."""
    first_year: int
    n_years: int
    subs: list
    centre: np.ndarray    # [S, 2] centre (row, col) on the unpadded HRES grid
    fc: np.ndarray        # [G, 10, Hp, Wp] float32 log1p(tp24) of the run issued on day g
    era_h: np.ndarray     # [G, 3, Hp, Wp] float32 tcwv, u850, v850 at g 00Z on the HRES grid
    static_h: np.ndarray  # [S, 4, WIN, WIN] float32 outline, orography, land-sea, in-domain
    domain: np.ndarray    # [Hp, Wp] float32, 1 inside the HRES domain
    maps: np.ndarray      # [G, 7, 37, 47] float32 tcwv, mslp, u850, v850, q850, z500, u200
    map_static: np.ndarray  # [2, 37, 47] float32 orography, land-sea mask

    @property
    def sub_index(self) -> dict:
        return {s: i for i, s in enumerate(self.subs)}

    # ---------------------------------------------------------------- build
    @classmethod
    def from_datasets(cls, hres: dict, era2d: dict, era3d: dict, static, weights: pd.DataFrame,
                      subs=None) -> "Grids":
        """Build from opened xarray datasets keyed by year. The constructor the
        loader, the tests and the audit's poisoning check all share."""
        years = sorted(hres)
        if years != list(range(years[0], years[0] + len(years))):
            raise ValueError(f"seasons must be consecutive, got {years}")
        subs = sorted(weights.subdivision_id.unique()) if subs is None else list(subs)
        h0 = hres[years[0]]
        lat, lon = h0.latitude.to_numpy().astype(float), h0.longitude.to_numpy().astype(float)
        elat = static.latitude.to_numpy().astype(float)
        elon = static.longitude.to_numpy().astype(float)
        A_lat, A_lon = bilinear_matrix(elat, lat), bilinear_matrix(elon, lon)

        def era_grid(da):  # any ERA5 field -> (..., lat, lon) float64
            return da.transpose(..., "latitude", "longitude").to_numpy().astype(np.float64)

        def on_hres(f):    # (..., 37, 47) -> (..., 45, 49), bilinear
            return np.einsum("ia,...ab,jb->...ij", A_lat, f, A_lon)

        fc, era_h, maps = [], [], []
        for y in years:
            h = hres[y]
            t = pd.to_datetime(h.time.to_numpy())
            if len(t) != DAYS or t[0] != pd.Timestamp(f"{y}-06-01") or \
                    int(h.sizes["prediction_timedelta"]) != N_LEADS:
                raise ValueError(f"HRES {y}: expected 122 issue days from 1 June x 10 leads")
            tp = h.tp24.transpose("time", "prediction_timedelta", "latitude", "longitude")
            fc.append(np.log1p(tp.to_numpy().astype(np.float64)))

            e2, e3 = era2d[y], era3d[y]
            et = pd.to_datetime(e2.time.to_numpy())
            want = pd.date_range(f"{y}-06-01", periods=DAYS, freq="D")  # 00Z of every issue day
            pos = pd.Index(et).get_indexer(want)
            pos3 = pd.Index(pd.to_datetime(e3.time.to_numpy())).get_indexer(want)
            if (pos < 0).any() or (pos3 < 0).any():
                raise ValueError(f"ERA5 {y}: an issue day has no 00Z analysis")
            lev = list(e3.level.to_numpy().astype(int))

            def lv(name, p):
                return era_grid(e3[name].isel(time=pos3, level=lev.index(p)))

            tcwv = era_grid(e2.total_column_water_vapour.isel(time=pos))
            mslp = era_grid(e2.mean_sea_level_pressure.isel(time=pos))
            u850, v850 = lv("u_component_of_wind", 850), lv("v_component_of_wind", 850)
            q850, z500 = lv("specific_humidity", 850), lv("geopotential", 500)
            u200 = lv("u_component_of_wind", 200)
            maps.append(np.stack([tcwv, mslp, u850, v850, q850, z500, u200], axis=1))
            era_h.append(on_hres(np.stack([tcwv, u850, v850], axis=1)))

        orog = era_grid(static.geopotential_at_surface)
        lsm = era_grid(static.land_sea_mask)
        domain = pad(np.ones((len(lat), len(lon))))
        outline = pad(outlines(weights, subs, lat, (len(lat), len(lon))))
        orog_h, lsm_h = pad(on_hres(orog)), pad(on_hres(lsm))
        centre = centres(weights, subs)
        static_h = np.zeros((len(subs), 4, WIN, WIN))
        for i, (r, c) in enumerate(centre):
            win = (slice(r, r + WIN), slice(c, c + WIN))  # padded coords: centre + PAD - PAD
            static_h[i] = np.stack([outline[i][win], orog_h[win] * domain[win],
                                    lsm_h[win] * domain[win], domain[win]])
        def f32(a):  # C order always: reductions over these arrays then sum in one order
            return np.ascontiguousarray(a, dtype=np.float32)

        return cls(first_year=years[0], n_years=len(years), subs=subs, centre=centre,
                   fc=f32(pad(np.concatenate(fc))),
                   era_h=f32(pad(np.concatenate(era_h)) * domain),
                   static_h=f32(static_h), domain=f32(domain),
                   maps=f32(np.concatenate(maps)),
                   map_static=f32(np.stack([orog, lsm])))

    @classmethod
    def load(cls, years=config.ALL_YEARS) -> "Grids":
        import xarray as xr

        def opened(path):
            with xr.open_dataset(path, decode_timedelta=True) as ds:
                return ds.load()

        hres = {y: opened(HRES_DIR / f"hres_tp24_india_{y}.nc") for y in years}
        era2d = {y: opened(ERA5_DIR / f"era5_2d_{y}.nc") for y in years}
        era3d = {y: opened(ERA5_DIR / f"era5_3d_{y}.nc") for y in years}
        static = opened(ERA5_DIR / "era5_static.nc")
        return cls.from_datasets(hres, era2d, era3d, static, pd.read_parquet(WEIGHTS))

    # ---------------------------------------------------------------- rows
    def row_index(self, df: pd.DataFrame):
        """Per row: global issue day g, lead index l (0..9) and subdivision index s."""
        t = pd.to_datetime(df.init_date).dt.normalize()
        year = t.dt.year.to_numpy()
        june1 = pd.to_datetime(pd.DataFrame({"year": t.dt.year, "month": 6, "day": 1}))
        day = (t - june1).dt.days.to_numpy()
        if ((year < self.first_year) | (year >= self.first_year + self.n_years)).any() or \
                ((day < 0) | (day >= DAYS)).any():
            raise ValueError("a row's issue day is outside the gridded seasons")
        g = (year - self.first_year) * DAYS + day
        l_ = df.lead_day.to_numpy(np.int64) - 1
        if ((l_ < 0) | (l_ >= N_LEADS)).any():
            raise ValueError("lead_day outside 1..10")
        index = self.sub_index
        s = np.array([index[x] for x in df.subdivision_id], dtype=np.int64)
        return g.astype(np.int64), l_, s

    @staticmethod
    def previous(g: np.ndarray, l_: np.ndarray):
        """The run issued on t - 1 at lead L + 1 verifies on the same day. It is
        absent at Day 10 and on 1 June (no May run in the store)."""
        ok = (g % DAYS > 0) & (l_ < N_LEADS - 1)
        return np.where(ok, g - 1, g), np.where(ok, l_ + 1, l_), ok

    def windows_raw(self, g, l_, s):
        """[n, 10, WIN, WIN] unscaled windows and the [n, 10, WIN, WIN] presence mask."""
        g, l_, s = (np.asarray(a, np.int64) for a in (g, l_, s))
        rows = self.centre[s, 0][:, None, None] + np.arange(WIN)[None, :, None]
        cols = self.centre[s, 1][:, None, None] + np.arange(WIN)[None, None, :]
        dom = self.domain[rows, cols]
        gp, lp, ok = self.previous(g, l_)
        okb = ok[:, None, None].astype(np.float32)
        this = self.fc[g[:, None, None], l_[:, None, None], rows, cols]
        prev = self.fc[gp[:, None, None], lp[:, None, None], rows, cols] * okb
        era = self.era_h[g[:, None, None, None], np.arange(3)[None, :, None, None],
                         rows[:, None], cols[:, None]]
        st = self.static_h[s]
        raw = np.concatenate([this[:, None], prev[:, None], (okb * dom)[:, None],
                              st[:, 0:1], st[:, 1:2], st[:, 2:3], st[:, 3:4], era], axis=1)
        present = np.repeat(dom[:, None] > 0, N_WIN, axis=1)
        present[:, FC_PREV] &= ok[:, None, None]
        return raw, present

    def maps_raw(self, g) -> np.ndarray:
        """[n, 9, 37, 47] unscaled maps for issue days g: 7 analysed at g 00Z, 2 static."""
        g = np.asarray(g, np.int64)
        st = np.broadcast_to(self.map_static, (len(g),) + self.map_static.shape)
        return np.concatenate([self.maps[g], st], axis=1)


# ------------------------------------------------------------------ scaling
@dataclass
class WindowScaler:
    """Per standardised channel: mean and sd of the present entries of the
    training rows' windows. Absent entries are 0 after scaling."""
    mean: np.ndarray   # [N_WIN], 0 for unscaled channels
    std: np.ndarray    # [N_WIN], 1 for unscaled channels

    @classmethod
    def fit(cls, grids: Grids, g, l_, s, chunk: int = 8192) -> "WindowScaler":
        n = np.zeros(N_WIN)
        tot = np.zeros(N_WIN)
        sq = np.zeros(N_WIN)
        for i in range(0, len(g), chunk):
            raw, pres = grids.windows_raw(g[i:i + chunk], l_[i:i + chunk], s[i:i + chunk])
            x = np.where(pres, raw.astype(np.float64), 0.0)
            n += pres.sum(axis=(0, 2, 3))
            tot += x.sum(axis=(0, 2, 3))
            sq += (x ** 2).sum(axis=(0, 2, 3))
        mean, std = np.zeros(N_WIN), np.ones(N_WIN)
        for c in STANDARDISED:
            if n[c] > 0:
                mean[c] = tot[c] / n[c]
                var = max(sq[c] / n[c] - mean[c] ** 2, 0.0)
                std[c] = np.sqrt(var) if var > 0 else 1.0
        return cls(mean, std)

    def transform(self, raw: np.ndarray, present: np.ndarray) -> np.ndarray:
        out = raw.astype(np.float64).copy()
        for c in STANDARDISED:
            out[:, c] = np.where(present[:, c], (raw[:, c] - self.mean[c]) / self.std[c], 0.0)
        return out.astype(np.float32)


@dataclass
class MapScaler:
    """Dynamic channels: each cell's mean over the training issue days removed,
    then divided by the channel's sd of those anomalies. Orography standardised;
    the land-sea mask unscaled."""
    cell_mean: np.ndarray  # [7, 37, 47]
    sd: np.ndarray         # [7]
    orog_mean: float
    orog_sd: float

    @classmethod
    def fit(cls, grids: Grids, train_days) -> "MapScaler":
        days = np.unique(np.asarray(train_days, np.int64))
        m = grids.maps[days].astype(np.float64)
        cell_mean = m.mean(axis=0)
        sd = (m - cell_mean).std(axis=(0, 2, 3))
        sd = np.where(sd > 0, sd, 1.0)
        orog = grids.map_static[0].astype(np.float64)
        osd = float(orog.std())
        return cls(cell_mean, sd, float(orog.mean()), osd if osd > 0 else 1.0)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """[n, 9, 37, 47] unscaled -> scaled float32."""
        out = raw.astype(np.float64).copy()
        out[:, :N_MAP_DYN] = (out[:, :N_MAP_DYN] - self.cell_mean) / self.sd[:, None, None]
        out[:, N_MAP_DYN] = (out[:, N_MAP_DYN] - self.orog_mean) / self.orog_sd
        return out.astype(np.float32)

    def all_maps(self, grids: Grids) -> np.ndarray:
        """[G, 9, 37, 47]: every issue day's map, scaled with this fold's training days."""
        return self.transform(grids.maps_raw(np.arange(len(grids.maps))))


_DEFAULT = None


def default_grids() -> Grids:
    """The real store, loaded once per process."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Grids.load()
    return _DEFAULT
