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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(config.ROOT)} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
