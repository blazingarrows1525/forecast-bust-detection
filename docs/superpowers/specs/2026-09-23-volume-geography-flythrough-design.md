# Volume geography + fly-through — design

**Date:** 2026-09-23
**Status:** implemented. Approved in chat in four sections; §10 records where the build departed from them and why.
**Builds on:** [`2026-09-20-volumetric-risk-field-design.md`](2026-09-20-volumetric-risk-field-design.md)
**Pages touched:** `web/volume.html`, `web/command.html`
**Build step touched:** `scripts/precompute_voxel_grid.py`

---

## 1. Why

The volume view is correct but disorienting. Coloured blocks float in a
wireframe box and nothing tells the viewer it is India. The request was
"deeper 3-D", for **both** a viewer watching (a guided fly-through) and a
viewer exploring (free orbit, slice, hover), sharing one scene.

## 2. A defect found before any design work: the volume is mirrored

Measured in the browser before this work started, by sampling the pick pass
across the default view:

| region | mean screen x (1024 px wide) |
|---|---|
| Arunachal Pradesh (≈ 94°E) | 251 |
| Assam & Meghalaya (≈ 92°E) | 332 |
| Gujarat Region (≈ 72°E) | 704 |
| Saurashtra & Kutch (≈ 70°E) | 746 |

**East renders on the left.** North–south is correct, so this is a reflection,
not a rotation: the shipped volume shows a mirror image of India.

Cause: the shader maps `+x → east` and `+z → north`, and the default camera
sits at `−z` (south) looking north. For a camera at `−z` looking towards `+z`,
three.js's right-hand vector is `−x`, so east lands on the left.

Why nothing caught it: the pick pass shares the display pass's swizzle, so the
readout always named the cell under the cursor correctly. The picture and the
pick agreed with each other and both were mirrored. With no geography in the
scene, there was nothing to disagree with.

**Fix:** north becomes `−z`. Both shaders map local space to the grid through
one `toGrid` (latitude runs along `0.5 − z`) in a shared prelude, one
JavaScript function `lonLatToWorld` is the only place geography becomes scene
coordinates, and the default camera moves to `+z` (the south side).
`test_north_is_minus_z_everywhere` pins all three. Browser check after the fix:
Assam at mean screen-x 694, Gujarat at 320.

## 3. Geography: approach C, with its mismatch bounded

Two geographies, chosen explicitly over the single-source alternative:

- **Floor:** a smooth vector outline of India plus a dark land fill, on a ground
  plane just below Day 1. It reads as the map under the volume.
- **Inside the volume:** staircase edges derived from the model's own 0.25° grid,
  drawn on the sliced lead layer only.

"Outline" and not "coastline": the union of the 36 subdivisions includes land
borders, which are not coast. The field is named `outline` and the legend says
so.

### 3.1 The mismatch, handled rather than hidden

The smooth outline is finer than the grid, so the two will not coincide at the
coast. Three measures keep that honest:

1. **Placement.** The outline sits on a ground plane *below* the data, so an
   offset reads as perspective rather than risk leaking into the sea.
2. **Measurement.** A test bounds the disagreement in both directions (outline
   vertex → nearest covered cell; grid edge cell → nearest outline vertex). The
   bounds are set from the measured values with small slack, and the measured
   maxima are recorded in D-024 rather than only a pass/fail.
3. **Labelling.** Legend: *"Outline smoothed for orientation. Edges inside the
   volume are the model's 0.25° cells."*

### 3.2 Build-time outline

`precompute_voxel_grid.py` gains an `outline` field in `voxel_grid.json`:
union of the 36 subdivisions → close slivers → exterior rings only → simplify
→ reproject to lon/lat → round. One script writes both geographies, so the
agreement test reads them from one artifact. No new endpoint, no runtime geo
stack (D-020 holds).

### 3.3 In-volume edges, face-aware, in the spare A channel

The texture is RGBA8: R probability, G status, B region index, and **A was
unused**. It now carries eight bits per cell, computed from `grid.ids`
(geography, independent of the risk data):

| bits 0–3 | a *different subdivision* lies across the east / west / north / south face |
|---|---|
| **bits 4–7** | **no subdivision lies across that face (outline)** |

Edges are evaluated **analytically**: one ray–plane intersection per pixel at
the top face of the sliced layer, one texel fetch there, and a line drawn at
the flagged faces using `fwidth` for a stable width in pixels. That point is
composited into the march at its correct depth. Sampling thin lines inside the
march itself would alias badly, because the march steps about one voxel at a
time.

## 4. Exploration tools

**One view-state object** — `{ theta, phi, r, target, slice, hover, focus }`.
The explore tools and the fly-through both write to it and the renderer only
reads it. `target` is new: the camera used to always look at the origin.

- **Slice** (`uSlice`, an `int` uniform, 0 = off, 1–10 = lead day). The sliced
  layer renders at full density, every other layer dims to near-invisible, and
  edges draw on the sliced layer. It snaps to whole lead days. The camera may
  move continuously; the data never shows a Day 3.5.
- **Hover focus.** The pick pass already names the region under the cursor via
  the B channel. Focus **dims every other region**; it does not brighten the
  hovered one.
  *Refinement to the approved design:* the design said "brighten". Brightening
  pushes a colour towards white, and on viridis lighter means further up the
  ramp, so a brightened region would read as higher risk than it is. Dimming
  the rest leaves the hovered region's colours exactly true.
- **Pick parity.** With a slice active, the pick pass can only hit the sliced
  layer. Otherwise a click aimed at Day 4 could name a nearly invisible Day 9
  voxel in front of it, and "what you click is what you saw" would fail with no
  visible symptom.

## 5. Fly-through

Everything it shows comes from the date on screen:

- **Hold target:** `GET /api/review-queue?init_date=…&top=1`.
- **Aim:** that region's lon/lat through `lonLatToWorld`.
- **Captions:** built from `/api/risk-cube` rows. No case number appears in the
  page source.

```
start   slice 10, camera high and wide      "Issued <init> · ten days out"
descend slice steps one whole lead at a time; camera eases continuously,
        turns slightly towards the target, look-at moves onto its centroid
hold    focus on target region              "<region> · Day <n> · valid <date> — <p>% · <tier>"
                                            (refused: "not scored" + 23.4% vs 3.4%)
finish  slice steps to 1, then off; overview
        outcome caption, only if truth exists "Observed <obs> mm vs forecast <fcst> mm — BUSTED / held"
release control returns; the scene stays where it stopped
```

- The camera turns at most ±0.3 rad from due south, so north stays roughly up
  on screen and the map is always readable.
- Any drag, wheel or key cancels at once and leaves state where it is.
  `Space` plays and stops.
- Changing the date cancels, so captions from one date cannot survive onto
  another.
- `prefers-reduced-motion` → three cuts (start, hold, finish), about 3 s each,
  with no camera interpolation.
- Captions sit in an `aria-live="polite"` region.
- It never starts on its own.

**Keys:** `←/→` slice, `S` slice off and back on, `Space` fly-through, `Esc`
stop and clear focus.

## 6. `command.html`, the WebGL1 fallback

It stays, because the volume needs WebGL2. Brought into line with the
dashboard:

- The five risk stops become the dashboard's viridis stops exactly. The old
  ramp had protanopia ΔE 10.0 between its safest and most dangerous bands.
- Refused columns: an achromatic striped texture instead of purple wireframe.
  Refused text stops being coloured purple.
- The truth colour matches the dashboard's `--truth-bust`.
- The legend copy stops saying "purple".

## 7. Failure handling

| situation | behaviour |
|---|---|
| No WebGL2 | Existing message, plus a link to `command.html` |
| `voxel_grid.json` without `outline` | No floor, legend note hidden, no error |
| Review queue empty or failing | The flight descends 10 → 1 with no hold |
| Hold target refused | Refusal caption, never a number |
| No truth for the target | No outcome caption |
| Date changed mid-flight | Flight cancels, caption cleared |
| Hover over no data | No focus |

## 8. Ruled out

- **Bloom / glow / `EffectComposer`.** Glow spreads a cell's colour into its
  neighbours, which is interpolation by another name.
- **Brightening as highlight.** See §4.
- **Rotating the flight freely.** A map seen from arbitrary azimuths stops
  being readable as a map.

## 9. Tests

File-level, because a shader cannot assert on itself:

1. The outline agrees with the grid in both directions within the measured bounds.
2. `uSlice` is declared `int` in both shaders.
3. The pick shader skips non-slice layers; both shaders use the new swizzle.
4. No `Bloom`, `UnrealBloomPass` or `EffectComposer` in `volume.html`.
5. No case numbers (`74.2`, `70.6`, `115.6`, `83.7`) in `volume.html`.
6. `prefers-reduced-motion` handled; captions in `aria-live`.
7. `command.html`'s five stops equal `index.html`'s; old green and red literals absent; no purple refusal.
8. The WebGL2 failure message links `command.html`.

Browser verification: handedness (Assam's mean x > Gujarat's), floor outline,
Day-4 slice with edges, hover focus, the hold frame, `command.html` in viridis,
an fps reading and a clean console.

## 10. Departures found during implementation

Each was a defect or a better-grounded choice uncovered by building and
looking. Full measurements are in `DECISIONS.md` D-024.

| approved | built | why |
|---|---|---|
| Keep the fixed-step march | Exact per-voxel traversal (DDA) in a shared `PRELUDE` | Slicing exposed banding. Jitter turned it into grain, the refusal medium's texture. |
| Keep the review "isosurface" | Review **boundary** line on the slice, flag from the real *p* | The shell was a ±0.006 band that tinted regions near-white, and 8-bit *p* disagreed with the readout. |
| Pick the first visible voxel | Pick the dominant contributor | The first-hit rule named a faint Day 9 layer while the eye was on the Day 3 mass. |
| `command.html`: palette + refusal | Also: camera moved to the south side | Its default view showed India upside down (a rotation, not a mirror). |
| Test bound "within one cell diagonal of a coast cell" | Two-way bound: 0.75 cells (outline→grid), 1.0 cell (grid→outline, to segments) | Measured as 0.70 and 0.94. Measuring to vertices overstated it as 1.52. |
