# Volumetric risk field — design

**Date:** 2026-09-20
**Status:** implemented (`web/volume.html`, `scripts/precompute_voxel_grid.py`)
**Scope:** one sub-project of a five-way decomposition (see §7)

---

## 1. Why

The bust-risk field is genuinely three-dimensional — longitude × latitude ×
lead day. `web/command.html` renders it as extruded columns, which is a 2.5-D
compromise that discards continuity along the lead axis and costs 343 draw
calls. This raymarches the field directly.

The project's stated intent for this work is a **technical playground**:
building hard things well is the point, and "is it useful" is explicitly
secondary to "is it well built". Correctness invariants still bind.

## 2. The principle everything else follows from

> **The render must never imply more resolution than the model has.**

Volume rendering conventionally wants trilinear interpolation. This field
cannot have it:

- Interpolating along **lead** invents a probability for "Day 3.5" that the
  model never computed.
- Interpolating **geographically** is worse. The prediction is constant within
  a subdivision, so linear filtering manufactures a gradient between Assam and
  Meghalaya that exists in no model output.

Both violate `FRONTEND_LOGIC.md` §2.3. So: `NearestFilter` on all three axes.
The volume is blocky, and the blockiness is the honest shape of the data — 36
irregular regions on a 0.25° grid, 10 discrete lead days.

This costs nothing that matters. Raymarching still buys accumulation along the
view ray, correct depth ordering, seeing *through* the field, and a distinct
optical medium for refusal. Smooth interpolation was never the value.

The invariant is guarded the same way the no-external-origin rule is — by
asserting on the file, in `test_volume_view_samples_nearest_on_every_axis`,
because a shader cannot assert on itself.

## 3. Architecture

**Server rasterises geometry once; the client assembles values per request.**

```
weights_129x135_*.parquet          exact EPSG:7755 polygon-cell overlap
  └─ scripts/precompute_voxel_grid.py      (build time, needs geopandas-era data)
       └─ data/interim/voxel_grid.json     21 KB, committed
            └─ GET /api/voxel-grid         static passthrough
                 └─ web/volume.html  ──┐
GET /api/risk-cube?init_date=… ─────────┴─→ RGBA8 Data3DTexture → raymarch
```

**Why the grid is a precompute.** Point-in-polygon over 36 multipolygons for
~15,000 cells, in JavaScript, on every page load, against *simplified*
geometry, would be slow and wrong in a way that matters — the simplified
outline is not the boundary the model was fitted on. Same reasoning as D-020.

**Why `weights_*.parquet` and not a fresh rasterisation.** That file already
holds the exact polygon-cell area overlap the feature pipeline itself uses, so
the rendered volume sits on precisely the grid the model was trained on rather
than a second grid that merely resembles it. A misaligned volume still looks
like India while placing every probability in the wrong cell; nothing in the
picture would reveal it. `test_grid_is_the_model_s_own_grid` asserts it.

**Cell assignment.** A 0.25° cell can overlap several subdivisions along a
boundary (1,227 of 5,107 covered cells do). The cell goes to the subdivision
holding the **largest area** within it. Blending would manufacture a value
neither model output contains.

## 4. Texture layout

| | |
|---|---|
| Dimensions | 125 (lon) × 123 (lat) × 10 (lead) |
| Format | `RGBAFormat` + `UnsignedByteType` → RGBA8, 615 KB |
| R | bust probability × 255 |
| G | status: 0 = scored, 1 = refused, 2 = no data |
| Filtering | `NearestFilter`, `ClampToEdgeWrapping`, all axes |

**`RGFormat` was the natural choice and does not work.** three.js r128 does not
map it to a sized internal format (`RG8`) for `texImage3D`, and WebGL2 rejects
the unsized form — verified in-browser: `RGBAFormat` → GL error 0,
`RedFormat` → 0, `RGFormat` → 1281 (`INVALID_VALUE`). RGBA8 costs ~300 KB more,
which is irrelevant here.

**Every voxel defaults to status 2 (no data), never to probability 0.** A zero
renders as a valid low reading, which is the one thing a missing value must
never look like. Island territories absent from the risk cube
(`ANDAMAN_NICOBAR`, `LAKSHADWEEP`) exercise this on every date.

## 5. Shader

Local box axes are `(x = lon, y = lead, z = lat)`; the texture is
`(u = lon, v = lat, w = lead)`, so the sample swizzles `local.zy`.

- **Geometry is a unit box scaled by the mesh.** `hitBox` intersects a box of
  half-extent 0.5 in local space; baking size into `BoxGeometry` puts every
  vertex far outside it and almost every ray misses. (This was the first bug —
  the volume rendered entirely black.)
- **Step count is `max(nVoxels · |dir|)`, not `min(n / |dir|)`.** The latter
  undersamples a diagonal ray roughly 4× and drops whole slabs, which on this
  data means silently hiding risk. (Second bug.)
- **Opacity is Beer–Lambert**, `1 − exp(−σ·dt)`, so the Density control means
  the same thing from every viewing angle rather than varying with how many
  samples the ray happened to take.
- **Colour ramp is viridis** via polynomial fit: perceptually uniform and
  colour-blind safe, which matters because this ramp encodes risk.
- **Isosurface** — a thin bright shell where the field crosses
  `REVIEW_THRESHOLD = 0.0909`, the cost-optimal threshold from
  `quality/escalation.py`. Not a free parameter.
- **Ambient floor** gives every *assessed* cell a faint presence at
  probability ≈ 0. This is not decoration: "the model looked and returned a low
  number" is different information from "nothing here", and without it the
  landmass vanishes and all orientation is lost.

### Refusal

`OUT_OF_DISTRIBUTION` renders as **dense achromatic noise** — high opacity, no
hue. Loud enough to draw the eye, and impossible to read as a point on the risk
ramp because it has no hue at all.

This is the encoding requirement from `FRONTEND_LOGIC.md` §2.1 solved as a
shader problem rather than a 2-D hatch. On the 2022 test year refused days
busted 23.4% of the time against 3.4% for scored ones, so a refusal is
elevated, unknown risk — never quieter than a low reading.

Observed consequence, on `2022-06-21` (18,755 refused voxels): the refusal
medium **occludes the scored field behind it**. That is correct — you cannot
see through "we don't know" — and it is why the Refusals toggle exists.

## 6. Measured

| | columns (`command.html`) | volume (`volume.html`) |
|---|---|---|
| Draw calls | 343 | **1** (3 with the frame) |
| Frame rate | — | **150–199 fps** |
| Payload | GeoJSON + cube | 21 KB grid + cube |
| GPU memory | geometry | 615 KB texture |

## 7. Out of scope

This is sub-project **C** of five. Not built here: the narrative landing page
(A), dashboard upgrades — lead scrub, dark mode, keyboard, override UI (B),
online enhancement including satellite imagery (D), deployment and
accessibility audit (E).

Also deliberately not done:

- **three.js upgrade.** r128 already ships `DataTexture3D`,
  `RawShaderMaterial`, `GLSL3`, `RedFormat` and `NearestFilter`, so the
  volumetric work needed no new bundle and no API migration.
- **GPU picking.** The volume has no per-region readout yet; the `#readout`
  panel is scaffolded and unused.
- **WebGL1 fallback.** `sampler3D` requires WebGL2. The page detects and
  explains rather than degrading silently; the 2-D dashboard and the column
  view are unaffected.

## 8. Contract conflicts surfaced, not resolved

The originating request asked for live satellite imagery, real-time data and
AWS deployment. Recorded here because none are resolved by this sub-project:

1. **Live imagery vs. offline.** Mutually exclusive in one page. CI now
   enforces zero external origins across *all* of `web/*.html`, widened from
   `index.html` only.
2. **"Real-time" vs. a 2016–2022 archive.** There is no real-time weather in
   this system; `mode=live` returns `STALE` by design. "Real-time" can honestly
   mean system-status polling and nothing more.
3. **AWS.** D-021 withdrew it and the stated constraint was "funds over".
4. **Forecast Analyst interface.** Free-form narration stays off — measured
   fabrication rate 1-in-4 on the better local model (D-019 addendum 3).
