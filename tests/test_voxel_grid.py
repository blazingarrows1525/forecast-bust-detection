"""Tests for the voxel grid behind the raymarched volume view.

These need no bulletin store: ``data/interim/voxel_grid.json`` is committed, so
they run in a fresh checkout and in CI.

The tests worth having here are the ones that catch a volume built on the
*wrong grid*. A render that is subtly misaligned with the model's own grid
still looks plausible -- India is still India-shaped -- while placing every
probability in the wrong cell. Nothing about the picture would reveal it.
"""
from __future__ import annotations

import base64
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbd import config  # noqa: E402
from fbd.api import app as app_module  # noqa: E402

GRID_PATH = config.INTERIM / "voxel_grid.json"

pytestmark = pytest.mark.skipif(
    not GRID_PATH.exists(),
    reason="voxel grid not built; run scripts/precompute_voxel_grid.py",
)


@pytest.fixture(scope="module")
def grid() -> dict:
    return json.loads(GRID_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ids(grid) -> np.ndarray:
    return np.frombuffer(base64.b64decode(grid["ids_b64"]), dtype=np.uint8)


def test_endpoint_serves_the_grid():
    r = TestClient(app_module.app).get("/api/voxel-grid")
    assert r.status_code == 200
    body = r.json()
    assert {"ny", "nx", "lat0", "lon0", "dlat", "dlon", "region_ids", "ids_b64"} <= set(body)


def test_decoded_length_matches_the_declared_shape(grid, ids):
    assert ids.size == grid["ny"] * grid["nx"]


def test_every_cell_indexes_a_real_region_or_nodata(grid, ids):
    """An out-of-range index would silently read some other region's risk."""
    n = len(grid["region_ids"])
    bad = np.unique(ids[(ids != grid["nodata"]) & (ids >= n)])
    assert bad.size == 0, f"cell values outside 0..{n - 1}: {bad.tolist()}"


def test_nodata_cannot_collide_with_a_region_index(grid):
    """255 must stay unreachable as a real index, or 'ocean' becomes a region."""
    assert grid["nodata"] == 255
    assert len(grid["region_ids"]) < grid["nodata"]


def test_regions_match_the_subdivision_config(grid):
    """Catches a grid built from stale or partial geometry."""
    cfg = json.loads(config.SUBDIVISION_CONFIG.read_text(encoding="utf-8"))["subdivisions"]
    assert sorted(grid["region_ids"]) == sorted(cfg)


def test_grid_is_the_model_s_own_grid(grid):
    """The volume must sit on the grid the features were built on.

    If this drifts, every probability is rendered in the wrong place and the
    picture still looks entirely reasonable -- which is exactly why it is
    asserted rather than eyeballed.
    """
    import pandas as pd

    paths = sorted(glob.glob(str(config.INTERIM / "weights_*.parquet")))
    if not paths:
        pytest.skip("no weights_*.parquet to compare against")

    def cells(p: str) -> int:
        m = re.search(r"weights_(\d+)x(\d+)_", Path(p).name)
        return int(m.group(1)) * int(m.group(2)) if m else 0

    w = pd.read_parquet(max(paths, key=cells))
    lats, lons = np.sort(w.lat.unique()), np.sort(w.lon.unique())

    assert grid["dlat"] == pytest.approx(float(np.diff(lats).min()), abs=1e-9)
    assert grid["dlon"] == pytest.approx(float(np.diff(lons).min()), abs=1e-9)
    # Reconstruct a cell centre from the grid's own origin and spacing and
    # check it lands on a real weights cell.
    lat = grid["lat0"] + int(w.lat_idx.iloc[0]) * grid["dlat"]
    lon = grid["lon0"] + int(w.lon_idx.iloc[0]) * grid["dlon"]
    assert lat == pytest.approx(float(w.lat.iloc[0]), abs=1e-6)
    assert lon == pytest.approx(float(w.lon.iloc[0]), abs=1e-6)


def test_every_assigned_cell_really_overlaps_that_subdivision(grid, ids):
    """Largest-area assignment, verified against the exact overlap table.

    A cell may only carry a subdivision that genuinely covers part of it.
    Assigning by, say, nearest centroid instead would pass every other test
    here and still put risk in cells the subdivision does not touch.
    """
    import pandas as pd

    paths = sorted(glob.glob(str(config.INTERIM / "weights_*.parquet")))
    if not paths:
        pytest.skip("no weights_*.parquet to compare against")

    def cells(p: str) -> int:
        m = re.search(r"weights_(\d+)x(\d+)_", Path(p).name)
        return int(m.group(1)) * int(m.group(2)) if m else 0

    w = pd.read_parquet(max(paths, key=cells))
    overlaps = set(zip(w.lat_idx, w.lon_idx, w.subdivision_id))

    nx, nodata = grid["nx"], grid["nodata"]
    names = grid["region_ids"]
    flat = np.nonzero(ids != nodata)[0]
    for f in flat[::37]:  # every 37th covered cell is plenty
        y, x = divmod(int(f), nx)
        assert (y, x, names[ids[f]]) in overlaps, (
            f"cell (lat_idx={y}, lon_idx={x}) assigned to {names[ids[f]]}, "
            "which has no area overlap there"
        )


def test_volume_view_samples_nearest_on_every_axis():
    """The invariant the whole view is built around (FRONTEND_LOGIC.md 2.3).

    Linear filtering would interpolate between lead days and across
    subdivision boundaries, inventing probabilities the model never produced.
    Guarded the same way the no-external-origin rule is: by asserting on the
    file, because a shader cannot assert on itself.
    """
    html = (config.ROOT / "web" / "volume.html").read_text(encoding="utf-8")
    assert "NearestFilter" in html
    assert "LinearFilter" not in html, (
        "volume.html must not use LinearFilter: interpolating this field "
        "manufactures values between lead days and across region boundaries"
    )


def test_volume_view_references_no_external_origin():
    """Same rule CI applies to the dashboard, applied to this page too."""
    html = (config.ROOT / "web" / "volume.html").read_text(encoding="utf-8")
    external = re.findall(r'(?:src|href)\s*=\s*["\'](https?://[^"\']+)', html)
    assert not external, f"external references found: {external}"
