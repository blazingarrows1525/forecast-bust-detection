"""Rasterise the 36 subdivisions onto the model's own grid, once, at build time.

Why this is a precompute and not client-side work
-------------------------------------------------
The volumetric view needs to know, for every cell of a lat/lon grid, which
subdivision that cell belongs to. Doing that in the browser would mean
point-in-polygon over 36 multipolygons for ~17,000 cells on every page load,
in JavaScript, against *simplified* geometry -- slow, and wrong in a way that
matters: the simplified outline is not the boundary the model was fitted on.

It also would not match. ``data/interim/weights_*.parquet`` already holds the
**exact polygon-cell area overlap in EPSG:7755** that the feature pipeline
itself uses (LOGIC.md sec 3). Reusing it means the rendered volume is
voxelised on precisely the grid the model was trained on, rather than on a
second grid that merely looks similar. Same principle as D-020: geometry never
changes at runtime, so computing it per request buys nothing.

Run:  PYTHONPATH=src python scripts/precompute_voxel_grid.py
Out:  data/interim/voxel_grid.json   (~24 KB)

Cell assignment
---------------
A 0.25-degree cell can overlap several subdivisions along a boundary. The cell
is assigned to the subdivision holding the **largest area** within it, which is
the only assignment that does not invent a blend. Blending two subdivisions'
probabilities in a boundary cell would manufacture a value neither model output
contains -- the thing FRONTEND_LOGIC.md sec 2.3 forbids.

The outline (a second geography, on purpose)
--------------------------------------------
The payload also carries ``outline``: India's outer boundary as smooth
polylines, drawn on the ground plane under the volume for orientation. It is
deliberately finer than the grid, so it will not coincide with the grid's own
staircase edge at the coast. That mismatch is accepted, bounded and labelled
rather than hidden: tests/test_voxel_grid.py measures it in both directions,
and the page's legend says which geography is which. See D-024.

Written by this script, not a separate one, so both geographies come out of
one build step and the agreement test reads them from one artifact.

"Outline", not "coastline": the union of 36 subdivisions includes land borders.
"""
from __future__ import annotations

import base64
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fbd import config  # noqa: E402

OUT = config.INTERIM / "voxel_grid.json"

#: Value meaning "no subdivision covers this cell" -- ocean, or outside India.
#: 255 rather than 0 so that an off-by-one never silently reads as region 0.
NODATA = 255

# ----------------------------------------------------------------- outline
#: Metric CRS for every length below. Same one the weights were computed in.
OUTLINE_CRS = 7755
#: Per-subdivision simplification before the union. Unioning full-resolution
#: district geometry and buffering it takes minutes; the output is simplified
#: to ~1 km anyway, so 200 m first changes nothing visible and takes seconds.
PRESIMPLIFY_M = 200
#: Morphological close (buffer out, buffer back). Seals the slivers that
#: independent simplification opens between neighbouring subdivisions, and the
#: district-boundary gaps already in the source. Must exceed PRESIMPLIFY_M.
#: It also fills inlets narrower than ~5 km -- irrelevant on a 28 km grid.
CLOSE_M = 2500
#: Final simplification. ~9x finer than a 0.25-degree cell.
SIMPLIFY_M = 1000
#: Drop islands smaller than this. 1 km2, not larger: the Lakshadweep islands
#: are ~4 km2 each and the grid has cells for them, so the outline must too.
MIN_PART_KM2 = 1.0
#: ~100 m. Finer digits are payload with no visible effect.
DECIMALS = 3


def _pick_weights() -> Path:
    """The weights file on the model's own India grid (129x135 at 0.25 deg)."""
    paths = sorted(glob.glob(str(config.INTERIM / "weights_*.parquet")))
    if not paths:
        sys.exit(
            "no weights_*.parquet in data/interim.\n"
            "Run: PYTHONPATH=src python -m fbd.regions.build"
        )
    # Filename carries the grid: weights_<ny>x<nx>_<lat0>_<lat1>_<lon0>_<lon1>.
    # Take the finest, which is the one the features are built on.
    def cells(p: str) -> int:
        m = re.search(r"weights_(\d+)x(\d+)_", Path(p).name)
        return int(m.group(1)) * int(m.group(2)) if m else 0

    return Path(max(paths, key=cells))


def _outline() -> list[list[float]] | None:
    """India's outer boundary as rings of flat ``[lon, lat, lon, lat, ...]``.

    Exterior rings only. An interior ring would draw a line through the middle
    of land, and every cell under one is covered by the grid regardless.

    Returns None when the geo stack is missing, so the grid still builds; the
    page renders without a floor in that case rather than failing.
    """
    try:
        import geopandas as gpd
        import shapely
        from shapely.ops import unary_union
    except ImportError:
        print("[WARN] geopandas not installed -- building the grid without an outline")
        return None

    src = config.SUBDIVISION_GPKG
    if not src.exists():
        print(f"[WARN] {src.name} missing -- building the grid without an outline")
        return None

    g = gpd.read_file(src, layer="subdivisions").to_crs(OUTLINE_CRS)
    # Reprojection alone leaves a few self-intersections that make the union
    # raise; repair first.
    g["geometry"] = shapely.make_valid(g.geometry.values)
    g["geometry"] = g.geometry.simplify(PRESIMPLIFY_M)

    land = unary_union(list(g.geometry)).buffer(CLOSE_M).buffer(-CLOSE_M)
    parts = list(land.geoms) if land.geom_type == "MultiPolygon" else [land]
    kept = [p for p in parts if p.area / 1e6 >= MIN_PART_KM2]

    exteriors = gpd.GeoSeries(
        [shapely.Polygon(p.exterior).simplify(SIMPLIFY_M) for p in kept],
        crs=OUTLINE_CRS,
    ).to_crs(4326)

    rings = []
    for ring in exteriors:
        coords = np.round(np.asarray(ring.exterior.coords), DECIMALS)
        if len(coords) < 4:          # collapsed by simplification
            continue
        rings.append(coords.ravel().tolist())

    n_vertices = sum(len(r) // 2 for r in rings)
    print(f"outline: {len(parts)} land parts, {len(rings)} kept "
          f"(>= {MIN_PART_KM2} km2), {n_vertices:,} vertices")
    return rings


def main() -> int:
    src = _pick_weights()
    w = pd.read_parquet(src)
    print(f"weights: {src.name}  ({len(w):,} subdivision-cell overlaps)")

    # Grid geometry is recovered from the data, not the filename, so a
    # mislabelled file cannot silently shift the whole volume.
    lats = np.sort(w.lat.unique())
    lons = np.sort(w.lon.unique())
    dlat = float(np.round(np.diff(lats).min(), 6))
    dlon = float(np.round(np.diff(lons).min(), 6))

    lat0 = float(lats.min() - w.lat_idx.min() * dlat)
    lon0 = float(lons.min() - w.lon_idx.min() * dlon)
    ny = int(w.lat_idx.max()) + 1
    nx = int(w.lon_idx.max()) + 1
    print(f"grid: {ny} lat x {nx} lon at {dlat} deg, "
          f"origin lat {lat0:.4f} lon {lon0:.4f}")

    # Largest area wins the cell. Sorting then dropping duplicates keeps the
    # heaviest overlap per (lat_idx, lon_idx).
    best = (w.sort_values("weight_km2", ascending=False)
              .drop_duplicates(subset=["lat_idx", "lon_idx"], keep="first"))

    region_ids = sorted(w.subdivision_id.unique())
    if len(region_ids) > NODATA:
        sys.exit(f"{len(region_ids)} regions does not fit in a uint8 alongside NODATA")
    index_of = {rid: i for i, rid in enumerate(region_ids)}

    grid = np.full((ny, nx), NODATA, dtype=np.uint8)
    grid[best.lat_idx.to_numpy(), best.lon_idx.to_numpy()] = [
        index_of[r] for r in best.subdivision_id
    ]

    covered = int((grid != NODATA).sum())
    contested = int(len(w) - len(best))
    print(f"cells covered: {covered:,} of {ny * nx:,} ({covered / (ny * nx):.1%})")
    print(f"boundary cells resolved to a single subdivision: {contested:,}")

    payload = {
        "_doc": (
            "Region index per grid cell, for the volumetric view. Built from the "
            "exact EPSG:7755 polygon-cell overlap the feature pipeline uses, so "
            "the rendered volume sits on the model's own grid. Largest-area wins "
            "each cell; boundary cells are never blended."
        ),
        "source": src.name,
        "ny": ny, "nx": nx,
        "lat0": lat0, "lon0": lon0, "dlat": dlat, "dlon": dlon,
        "nodata": NODATA,
        "region_ids": region_ids,
        # Row-major, lat_idx outer, lon_idx inner. uint8, base64.
        "ids_b64": base64.b64encode(grid.tobytes()).decode("ascii"),
    }
    outline = _outline()
    if outline is not None:
        # Rings of flat [lon, lat, ...], EPSG:4326. Orientation only: finer
        # than the grid by design, and never used to place a value.
        payload["outline"] = outline
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(config.ROOT)} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
