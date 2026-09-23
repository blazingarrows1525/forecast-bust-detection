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


# ------------------------------------------------------------------ outline
# The floor carries a smooth outline that is deliberately finer than the grid
# (D-024, "approach C"). Two geographies that do not coincide are a standing
# risk: a future re-simplification or a CRS slip could drift them apart and the
# picture would still look like India. These tests put a number on "they
# agree" in both directions.
#
# Measured when written (2026-09-23): outline -> grid 0.70 cells worst case;
# grid -> outline 0.94 cells worst case (Little Andaman, and a Lakshadweep
# islet below the 1 km2 cut). The claim the UI makes is "never more than one
# cell apart", so that is what is enforced.
@pytest.fixture(scope="module")
def outline(grid) -> list[np.ndarray]:
    if "outline" not in grid:
        pytest.fail("voxel_grid.json has no outline; rebuild with "
                    "scripts/precompute_voxel_grid.py (needs geopandas)")
    return [np.asarray(r, dtype=float).reshape(-1, 2) for r in grid["outline"]]


def _cell_centres(grid, mask) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.c_[grid["lon0"] + xs * grid["dlon"], grid["lat0"] + ys * grid["dlat"]]


def _covered(grid, ids) -> np.ndarray:
    return ids.reshape(grid["ny"], grid["nx"]) != grid["nodata"]


def test_outline_rings_are_closed_and_in_lon_lat(grid, outline):
    for ring in outline:
        assert len(ring) >= 4, "a ring needs at least three distinct points"
        assert np.allclose(ring[0], ring[-1]), "ring is not closed"
        lon, lat = ring[:, 0], ring[:, 1]
        # Inside the grid's own extent: a CRS slip (metres, or lat/lon swapped)
        # lands far outside it.
        assert lon.min() >= grid["lon0"] - 1 and lon.max() <= grid["lon0"] + grid["nx"] * grid["dlon"] + 1
        assert lat.min() >= grid["lat0"] - 1 and lat.max() <= grid["lat0"] + grid["ny"] * grid["dlat"] + 1


def test_outline_never_runs_through_a_cell_the_grid_does_not_cover(grid, ids, outline):
    """Outline -> grid: every vertex lies inside (or on) a covered cell.

    A vertex inside a cell is at most half a diagonal (0.707 cells) from that
    cell's centre. The small slack absorbs 3-decimal rounding.
    """
    verts = np.concatenate(outline)
    centres = _cell_centres(grid, _covered(grid, ids))
    worst = 0.0
    for i in range(0, len(verts), 400):
        d = np.sqrt(((verts[i:i + 400, None, :] - centres[None]) ** 2).sum(-1)).min(1)
        worst = max(worst, float(d.max()))
    cells = worst / grid["dlon"]
    assert cells <= 0.75, (
        f"an outline vertex is {cells:.2f} cells from any covered cell: the floor "
        "is drawing land where the model has none"
    )


def test_grid_coast_is_never_more_than_one_cell_from_the_outline(grid, ids, outline):
    """Grid -> outline: every edge cell's centre is within one cell of the line.

    Measured against the line's segments, not its vertices -- a long straight
    stretch of coast has sparse vertices, and vertex distance would overstate
    the gap by half a segment.
    """
    cov = _covered(grid, ids)
    pad = np.pad(cov, 1, constant_values=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    pts = _cell_centres(grid, cov & ~interior)
    a = np.concatenate([r[:-1] for r in outline])
    ab = np.concatenate([r[1:] for r in outline]) - a
    len2 = np.maximum((ab ** 2).sum(1), 1e-12)
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / len2[None], 0, 1)
    d = np.sqrt(((a[None] + t[..., None] * ab[None] - pts[:, None, :]) ** 2).sum(-1)).min(1)
    cells = float(d.max()) / grid["dlon"]
    assert cells <= 1.0, (
        f"a grid edge cell is {cells:.2f} cells from the outline; the two "
        "geographies have drifted further apart than the legend says"
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


# ------------------------------------------------------------------ picking
# A volume has no surface, so picking is a second raymarch rather than a mesh
# intersection. That creates a failure mode with no visual symptom: the pick
# shader and the display shader drifting apart, so the readout names a
# different cell than the one under the cursor. Both guards below are
# file-level for the same reason the NEAREST one is -- a shader cannot assert
# on itself.
def _volume_html() -> str:
    return (config.ROOT / "web" / "volume.html").read_text(encoding="utf-8")


def _shader(html: str, name: str) -> str:
    """The body of one GLSL template literal, e.g. ``const FRAG = /* glsl */`...`;``."""
    start = html.index(f"const {name} = /* glsl */`")
    return html[start:html.index("`;", start + len(name) + 20)]


def test_pick_and_picture_march_the_same_way():
    """One traversal, included verbatim by both passes.

    This used to be two copies of a fixed-step march, kept in step by matching
    strings. Now both shaders are built from the same PRELUDE, so they cannot
    walk the grid differently -- and the pick's opacity rule is the picture's.
    """
    html = _volume_html()
    frag, pick = _shader(html, "FRAG"), _shader(html, "PICK_FRAG")
    assert frag.startswith("const FRAG = /* glsl */`${PRELUDE}")
    assert pick.startswith("const PICK_FRAG = /* glsl */`${PRELUDE}")
    prelude = _shader(html, "PRELUDE")
    # Each piece of traversal and opacity logic exists exactly once, in the prelude.
    for fn in ("vec2 hitBox(", "vec3 toGrid(", "void ddaInit(", "float ddaStep(",
               "float voxelAlpha(", "float hash("):
        assert html.count(fn) == 1, f"{fn} must be defined once, not per shader"
        assert fn in prelude, f"{fn} must live in the shared prelude"
    for body in (frag, pick):
        assert "ddaInit(" in body and "ddaStep(" in body and "voxelAlpha(" in body
    # The old fixed-step sampler is gone from both.
    assert "float steps =" not in html, "fixed-step sampling reintroduced"


def test_traversal_is_exact_per_voxel():
    """NEAREST, taken to its conclusion: one value per voxel, exact path length.

    Fixed-step sampling put 3 or 4 samples in a slab depending on ray phase
    (stripes); jitter turned that into grain, the refusal medium's texture.
    """
    html = _volume_html()
    frag = _shader(html, "FRAG")
    assert "texelFetch(uField, here, 0)" in frag, "the traversal must read the voxel it is in"
    assert "float seg = max(tExit - tCur, 0.0);" in frag
    assert "jitter" not in frag, "per-pixel jitter makes scored data look like refusal noise"


def test_north_is_minus_z_everywhere():
    """The handedness fix (D-024). The shipped volume showed India mirrored.

    Picture and pick shared a swizzle, so they agreed with each other while
    both being wrong; only geography in the scene could expose it. Pin the
    three places that define the convention.
    """
    html = _volume_html()
    assert "(0.5 - p.z) * size.y" in _shader(html, "PRELUDE"), "grid latitude must run along -z"
    assert html.count("function lonLatToWorld(") == 1
    assert "z: -(v - 0.5) * D" in html
    assert "theta: Math.PI / 2," in html, "default camera must sit on the south side"


def test_slice_is_a_whole_lead_day():
    """The camera may glide; the data may not. No Day 3.5."""
    html = _volume_html()
    assert html.count("uniform int   uSlice;") == 2, "uSlice must be an int in both passes"
    # \b: uSliceDim is legitimately a float.
    assert re.search(r"uniform\s+float\s+uSlice\b", html) is None
    assert "Math.round(n)" in html[html.index("function setSlice("):]


def test_pick_follows_the_slice_and_shares_its_uniform():
    """While slicing, only the sliced layer can be named by the readout."""
    html = _volume_html()
    pick = _shader(html, "PICK_FRAG")
    assert "bool candidate = uSlice == 0 || lead == uSlice;" in pick
    assert "uSlice: material.uniforms.uSlice," in html, (
        "the pick must share the display's uniform object, not copy its value"
    )


def test_pick_names_the_voxel_that_was_seen():
    """Dominant contributor, not first-with-any-opacity.

    "First" named a faint Day 9 layer while the eye was on the Day 3 mass.
    """
    pick = _shader(_volume_html(), "PICK_FRAG")
    assert "float w = (1.0 - acc) * a;" in pick
    assert "if (candidate && w > best)" in pick


def test_focus_and_slice_dim_opacity_never_colour():
    """Colour is the value. Dimming may touch alpha only.

    The approved design said "brighten the hovered region"; brightening moves
    a colour towards white, which on viridis reads as higher risk.
    """
    frag = _shader(_volume_html(), "FRAG")
    assert "vec4 src = vec4(col, a * keep);" in frag
    assert "mix(col, vec3(1.0)" not in frag, "whitening a value misstates it"


def test_review_flag_comes_from_the_real_probability():
    """At 8 bits, 0.092 stores as 0.0902 -- below the 0.0909 threshold.

    The readout decides REVIEW from the real value, so the shader must use a
    flag set from the real value too, or the two disagree on real cells
    (Sub-Himalayan WB, Day 3, 2022-06-14, is one).
    """
    html = _volume_html()
    assert "p >= REVIEW_THRESHOLD ? F_REVIEW : 0" in html
    frag = _shader(html, "FRAG")
    assert "(meG & 4) != 0" in frag and "(nG & 4) != 0" in frag
    assert "abs(prob - uThreshold)" not in frag, "the coincidence-band 'shell' is back"


def test_edges_come_from_the_model_grid():
    """In-volume edges are the model's own cell faces, from grid ids alone."""
    html = _volume_html()
    js = html[html.index("function computeEdgeBits("):]
    assert "b |= 16 << k" in js and "b |= 1 << k" in js
    assert "data[o + 3] = edgeBits[y * nx + x];" in html


def test_no_glow_postprocessing():
    """Glow spreads a cell's colour into its neighbours: interpolation by another name."""
    html = _volume_html()
    for banned in ("Bloom", "EffectComposer", "UnrealBloomPass"):
        assert banned not in html, f"{banned} would smear values across cells"


@pytest.mark.parametrize("number", ["74.2", "70.6", "115.6", "83.7", "99.4", "110.5"])
def test_flythrough_hardcodes_no_case_number(number: str):
    """Captions come from the date on screen. A written-in number stays
    plausible forever -- and the real 2022-06-14 top flag held, which a
    scripted story would have got wrong."""
    assert number not in _volume_html()


def test_flythrough_respects_reduced_motion_and_is_announced():
    html = _volume_html()
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in html
    assert "if (REDUCED.matches)" in html
    assert 'id="caption" class="panel" role="status" aria-live="polite"' in html


def test_webgl2_failure_points_at_the_fallback():
    html = _volume_html()
    msg = html[html.index("WebGL2 required"):html.index("return false;", html.index("WebGL2 required"))]
    assert 'href="command.html"' in msg


def test_readout_never_prints_a_probability_for_a_refused_cell():
    """The §2.1 invariant at the readout, not just in the render.

    A refused cell has no probability. Printing one -- or printing nothing,
    which reads as reassurance -- is the failure this whole project is about.
    """
    html = _volume_html()
    start = html.index('if (status === "OUT_OF_DISTRIBUTION")')
    branch = html[start:html.index("} else if", start)]
    assert '"not scored"' in branch
    assert "toFixed" not in branch, "no numeric probability may be rendered here"
    assert "REFUSED" in branch and "23.4%" in branch, (
        "a refusal must state that it is elevated risk, not merely absent"
    )
