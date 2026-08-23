"""Exact area-overlap weights between a lat/lon grid and the IMD subdivisions.

Why not ``regionmask``/centroid masking?  Because a centroid test assigns a cell
to a region only if the cell *centre* falls inside it.  At 0.7 deg that silently
gives zero cells to narrow coastal subdivisions -- Konkan & Goa, Coastal
Karnataka, Kerala -- which are precisely the heavy-rainfall regions this project
exists to serve.  Losing them would not raise an error; it would just quietly
drop the most important rows.

Instead we intersect each grid cell polygon with each subdivision polygon and
keep the true overlap area.  The result is a tidy weight table

    subdivision_id | lat | lon | weight_km2

which supports honest area-weighted means and, as a by-product, tells us how
many grid cells actually inform each subdivision (a data-quality signal we
surface in the API).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from fbd import config

# Equal-area projection for India (EPSG:7755, India NSF LCC).  Areas computed in
# a geographic CRS would be wrong by ~20% between Kerala and Kashmir.
EQUAL_AREA_CRS = 7755


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Infer cell edges from centre coordinates (assumes regular spacing)."""
    c = np.asarray(centres, dtype=float)
    step = np.median(np.diff(c))
    return np.concatenate([[c[0] - step / 2], c[:-1] + np.diff(c) / 2, [c[-1] + step / 2]])


def grid_cells(lats: np.ndarray, lons: np.ndarray) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of grid-cell rectangles for the given centres."""
    lat_e, lon_e = _cell_edges(lats), _cell_edges(lons)
    recs = []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            recs.append(
                {
                    "lat_idx": i,
                    "lon_idx": j,
                    "lat": float(la),
                    "lon": float(lo),
                    "geometry": box(
                        min(lon_e[j], lon_e[j + 1]),
                        min(lat_e[i], lat_e[i + 1]),
                        max(lon_e[j], lon_e[j + 1]),
                        max(lat_e[i], lat_e[i + 1]),
                    ),
                }
            )
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")


def _cache_key(lats: np.ndarray, lons: np.ndarray) -> str:
    return (
        f"weights_{len(lats)}x{len(lons)}"
        f"_{lats[0]:.4f}_{lats[-1]:.4f}_{lons[0]:.4f}_{lons[-1]:.4f}.parquet"
    )


def overlap_weights(
    lats: np.ndarray,
    lons: np.ndarray,
    subs: gpd.GeoDataFrame | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return tidy (subdivision_id, lat_idx, lon_idx, weight_km2) overlaps.

    The overlay costs ~40 s on the IMD grid, so results are cached per grid
    geometry.  The cache key encodes the grid shape and corner coordinates, so a
    different grid can never silently reuse another grid's weights.
    """
    cache = config.INTERIM / _cache_key(np.asarray(lats), np.asarray(lons))
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    if subs is None:
        subs = gpd.read_file(config.SUBDIVISION_GPKG, layer="subdivisions")

    cells = grid_cells(lats, lons)
    # Restrict to cells that touch India at all -- keeps the overlay small.
    hull = subs.geometry.union_all()
    cells = cells[cells.intersects(hull)].reset_index(drop=True)

    inter = gpd.overlay(
        cells, subs[["subdivision_id", "geometry"]], how="intersection", keep_geom_type=True
    )
    inter["weight_km2"] = inter.to_crs(EQUAL_AREA_CRS).area / 1e6
    inter = inter[inter.weight_km2 > 0]
    out = inter[
        ["subdivision_id", "lat_idx", "lon_idx", "lat", "lon", "weight_km2"]
    ].reset_index(drop=True)
    if use_cache:
        out.to_parquet(cache, index=False)
    return out


def weights_to_matrix(
    weights: pd.DataFrame, n_lat: int, n_lon: int, subdivision_ids: list[str]
) -> np.ndarray:
    """Dense (n_sub, n_lat, n_lon) weight matrix for fast vectorised means."""
    idx = {s: i for i, s in enumerate(subdivision_ids)}
    W = np.zeros((len(subdivision_ids), n_lat, n_lon), dtype=np.float32)
    for sid, li, lo, w in weights[
        ["subdivision_id", "lat_idx", "lon_idx", "weight_km2"]
    ].itertuples(index=False):
        W[idx[sid], li, lo] = w
    return W


def area_mean(
    field: np.ndarray, W: np.ndarray, max_nan_fraction: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted mean of ``field`` over each subdivision.

    Parameters
    ----------
    field : (..., n_lat, n_lon) array, may contain NaN (ocean / no-gauge cells).
    W     : (n_sub, n_lat, n_lon) area weights.

    Returns
    -------
    means : (..., n_sub) area-weighted means, NaN where coverage is too poor.
    covered : (..., n_sub) fraction of each subdivision's area that had data.

    Weights are renormalised over the valid cells only, so a subdivision that is
    half ocean is still averaged correctly over its land part -- but we also
    return the coverage fraction so the caller can reject thin coverage rather
    than quietly trusting a mean built from two cells.
    """
    max_nan_fraction = (
        config.MAX_NAN_FRACTION if max_nan_fraction is None else max_nan_fraction
    )
    lead_shape = field.shape[:-2]
    flat = field.reshape(-1, field.shape[-2], field.shape[-1])
    valid = np.isfinite(flat)
    filled = np.where(valid, flat, 0.0)

    # einsum over (time, lat, lon) x (sub, lat, lon) -> (time, sub)
    num = np.einsum("tij,sij->ts", filled, W, optimize=True)
    den = np.einsum("tij,sij->ts", valid.astype(np.float32), W, optimize=True)
    total = W.sum(axis=(1, 2))[None, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        means = num / den
        covered = den / total
    means[covered < (1.0 - max_nan_fraction)] = np.nan
    return means.reshape(*lead_shape, -1), covered.reshape(*lead_shape, -1)
