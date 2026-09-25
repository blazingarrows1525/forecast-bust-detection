# FRONTEND_BUILT.md — the logic of what shipped

Companion to [`FRONTEND_LOGIC.md`](../FRONTEND_LOGIC.md). That file is the
pre-build spec: what the interface must be true to. This file is the
retrospective: for each piece of frontend that shipped, the invariant it
embodies, the tension it resolves, and the bug the code carries a scar from.

If the two disagree, the spec wins and the code is wrong.

---

## 1. What shipped, at a glance

Five sub-projects were carved out of the original build spec. Four shipped;
the fifth is blocked on a decision only the user can make.

| | | pages | state |
|---|---|---|---|
| **A** | Narrative landing | `web/landing.html` (444 lines) | shipped |
| **B** | Dashboard upgrades | `web/index.html` (684 lines) | shipped |
| **C** | Volumetric risk field | `web/volume.html` (1,424 lines) | shipped |
| **C+** | Geography, slice, fly-through | `web/volume.html`, `web/command.html` | shipped (D-024) |
| **D** | Date-matched imagery | `web/index.html` (opt-in layer) | shipped |
| **E** | Deployment & hardening | — | blocked on hosting |

**279 tests pass.** The four pages ship with three vendored files (Leaflet
CSS/JS and `three.min.js`) and one 60 KB precomputed grid-and-outline file. No
fonts, no CDN, no analytics, no framework.

The volume view has two design docs:
[the original](superpowers/specs/2026-09-20-volumetric-risk-field-design.md) and
[geography + fly-through](superpowers/specs/2026-09-23-volume-geography-flythrough-design.md).
This file does not duplicate them.

---

## 2. Sub-project A — landing page (`web/landing.html`)

**Job:** tell the June 2022 story in a way a judge can read in under a minute,
without a single number that isn't live from the API.

**The invariant.** Every claim on the page reads live from the API: the
convergence chart from `/api/convergence`; the four headline numbers, the ENS
sentence and the refusal chart from `/api/metrics`. Anything missing renders
"—", never a number remembered from an old run. (This line used to say
everything came from `/api/convergence`. It was false: until `028bab8` the ENS
sentence, the stat cards and the refusal chart were hardcoded. The S1 audit
caught it, D-025.)

The line the page hinges on — *"the ensemble converged; this model went the
other way"* — is not hardcoded copy. It comes from a `GET` on 34 forecasts of
the same region-day, longest lead first, so the reader can watch the two
curves' actual shape. If the archive is regenerated tomorrow and the numbers
change, the page changes with them.

`landing.html:418` says so in the page itself: `Live from <code>/api/convergence</code>.`

**Endpoint added for it.** `GET /api/convergence?region_id=&valid_date=` at
`src/fbd/api/app.py:313`. One region, one target day, every forecast that ever
tried to predict it, sorted by lead descending. The endpoint exists so the
narrative could stop lying — an earlier draft baked the divergence values in.

**Motion earns its place** (per `FRONTEND_LOGIC.md` §6.2). The divergence
curve draws in on scroll via `stroke-dasharray` + `IntersectionObserver`, no
library. Everything else is static.

**Copy discipline.** The `may` and `may-not` list from `FRONTEND_LOGIC.md` §8
is applied literally, and the ENS sentence is chosen by the verdict in
`/api/metrics`, not written by hand. Since S1 (D-025) it reads "outranks a real
ensemble" over the 2022 held-out season, with the interval **+0.032 [+0.014,
+0.048]**, the date count and the year beside it. Since S1b (D-026) the
backtest follows it: outside that season the edge is not established, over
2019–2021 **+0.004 [−0.006, +0.013]**, and any year the ensemble won is named
(2019). The years, the interval and that sentence all come from the API; for a
while after S1b the page still said only "over the full held-out season",
which FRONTEND_LOGIC §8 no longer allowed, and the S1b audit caught it. Before
S1 it said the margin was not established, because the 40-date interval
contained zero. The page can state any of the three registered verdicts for
both tests, and a test checks that it can.

---

## 3. Sub-project B — dashboard upgrades (`web/index.html`)

Five pieces were added. Each carries one bug in the corner of the diff, kept
because it makes the invariant visible.

### 3.1 Viridis, not green → red

The old ramp was categorical green → red. A colour-vision audit measured
protanopia ΔE **10.0** between the safest and the most dangerous band — the
palette was actively unsafe on the primary user population for a met agency.

Replaced with a five-stop viridis ramp (`index.html:9`), the same ramp the
volume view uses, so the map and the volume tell the same colour story. The
comment above the CSS block records the ΔE that justified the change so a
future contributor doesn't "restore" the green.

### 3.2 Refusal is a pattern, not a hue

`FRONTEND_LOGIC.md` §2.1: a refused region must never render in the same
visual family as low risk. It renders as an SVG `<pattern id="oodHatch">` —
diagonal hatching, theme-aware, rebuilt when the theme flips
(`index.html:229, 245`).

The one measured accident this prevents: on the 2022 test year, refused days
bust **6.9× more often** than accepted ones. Grey would read as "nothing to
see". Hatch reads as "not scored".

### 3.3 Lead scrub, using the endpoint that was already there

`GET /api/risk-cube` returns the whole 34 × 10 field in one payload. The scrub
button walks Day 10 → Day 1 off that single payload, one animation frame per
lead, no round trips. This is what §6.2 of the spec called "worth building".

Space toggles it. Arrow keys step it. Escape stops it.

### 3.4 Keyboard, first-class

The forecaster has 45 minutes. Keys (`index.html:530`):

- `←` / `→` — lead
- `↑` / `↓` — date
- `space` — scrub
- `m` / `b` — model vs baseline
- `v` — truth overlay
- `t` — theme
- `i` — imagery
- `?` — help
- `/` — focus date picker
- `Esc` — clears selection, closes help, cancels the override form

No modifier keys — a duty forecaster is not going to remember Ctrl+Shift+M.

### 3.5 Override, the one write in the whole product

`POST /api/override` was already the only write endpoint. The dashboard now
has the form for it (`index.html:156`). The reason field is mandatory — that
is the point of an audit log. Dismissals and escalations are both stored;
they are immutable; every write carries a user and a reason so a decision can
be reconstructed after an event.

**Bug worth naming.** The status line was first added as a direct child of
the `#app` grid. It consumed the map's cell and squeezed the map to 380×25 px
while every functional check passed — tiles loaded, zero console errors,
right date. The fix was `grid-column: 1/-1`. The lesson: functional tests
cannot replace looking at the page.

### 3.6 Theme

`data-theme="dark"` is the default, `light` is stored in `localStorage`.
The hatch pattern bakes theme colours into its SVG, so `toggleTheme`
regenerates it (`index.html:474`). The `try/catch` around `localStorage` is
not defensive theatre — private windows throw.

---

## 4. Sub-project C — volumetric risk field (`web/volume.html`)

Full design in [`docs/superpowers/specs/2026-09-20-volumetric-risk-field-design.md`](superpowers/specs/2026-09-20-volumetric-risk-field-design.md).
Two things belong in *this* file because they generalise:

**The principle.** *The render must never imply more resolution than the model
has.* Volume rendering conventionally wants trilinear interpolation. This
field cannot have it — interpolating along lead invents a probability for
Day 3.5; interpolating geographically manufactures a gradient between Assam
and Meghalaya. `NearestFilter` on all three axes. The volume is blocky, and
the blockiness is the honest shape of the data.

**Refusal, again.** The 2-D dashboard solves it with an SVG hatch; the volume
solves it as a shader problem: dense achromatic noise, high opacity, no hue.
Impossible to read as a point on the viridis ramp because it has no hue at
all. `test_volume_view_samples_nearest_on_every_axis` guards the shader — a
shader cannot assert on itself.

**Endpoint added for it.** `GET /api/voxel-grid` at `src/fbd/api/app.py:243`.
Precomputed from `weights_129x135_*.parquet`: the same exact polygon-cell
overlap the feature pipeline itself uses, so the rendered volume sits on
precisely the grid the model was trained on rather than a second grid that
merely resembles it. `test_grid_is_the_model_s_own_grid` asserts it.

### 4.1 C+: geography, slice, hover, fly-through (D-024)

**It was a mirror image, and nothing could tell.** The shipped volume put east
on the left. Picture and pick shared one axis swizzle, so they agreed with each
other while both being wrong, and with no geography in the scene there was
nothing to disagree with. North is now `−z`, one `lonLatToWorld` is the only
coordinate conversion, and the camera sits on the south side. The column view
had a different defect, a camera on the north side (India upside down), fixed
the same way.

**Two geographies, bounded.** A smooth outline on a ground plane below Day 1
for orientation, and the model's own 0.25° cell faces (from the texture's spare
A channel) drawn on the sliced layer. Measured: never more than one cell apart
in either direction (0.70 and 0.94 cells), and the legend says so.

**Exact traversal, not sampling.** The field is constant per voxel, so each ray
walks the grid voxel by voxel and integrates exact path lengths. Fixed steps
striped; jitter turned the stripes into grain, and grain is refusal's
texture. Display and pick share one `PRELUDE`, so parity is structural.

**Slice to read, full volume to see mass.** In the full view a pixel blends
every value along its ray and can land off the ramp. The slice (an `int`, whole
lead days only) shows one day's true values, its grid edges, and the review
boundary.

**The review boundary replaced a fake isosurface.** The old shell lit voxels
within ±0.006 of 9.09%, which on this data meant whole regions, tinted
near-white (refusal's colour family). The boundary is now a line where flagged
meets unflagged, and the flag is set from the real probability because 8-bit
*p* disagreed with the readout (0.092 → 0.0902).

**Focus dims, never brightens.** Brightening moves a value up the viridis ramp.

**Pick names what you saw**: the dominant contributor, from the display's own
opacity rule, not the first voxel with any opacity (which named a faint Day 9
layer over a bright Day 3 mass).

**The fly-through is a script over the explore controls,** driven by
`/api/review-queue?top=1` and `/api/risk-cube`. It reports what happened,
including that 2022-06-14's top flag (Assam, Day 3, 74.2%) *held*, which a
scripted demo would have got wrong. Cancelled by any input or a date change;
three cuts under reduced motion; captions `aria-live`.

---

## 5. Sub-project D — date-matched satellite imagery

The build spec asked for live imagery and offline air-gap in one page. Those
are mutually exclusive if read literally.

**How it was reconciled.** Imagery is:

1. **Off by default** (`IMAGERY.enabled = false`, `index.html:314`).
2. Sourced from **one** allowlisted origin, NASA EOSDIS GIBS, listed by name
   in the CI air-gap gate (`.github/workflows/ci.yml:152`).
3. Requested at the **valid date of the lead on screen**, never the clock —
   the tile URL uses `date = r.valid_date` (`index.html:344`). Init 2022-06-14
   + Day 4 requests 2022-06-17, the Assam floods day.
4. Auto-disabled after four consecutive tile errors (`index.html:351`).

**"Live" would have been the wrong answer.** There is no real-time weather in
this system. Overlaying today's satellite on 2022 risk would have been the
"fake live imagery" the spec explicitly forbade. Date-matched to the lead is
more useful and cannot mislead about which moment it shows.

**The CI gate that refused to be evaded.** The old check grepped `src`/`href`.
A Leaflet tile URL is built in JavaScript — it would have slipped past
silently and left the guard green. The new gate
(`.github/workflows/ci.yml:146`) scans every page for any `https?://…` origin
against a documented allowlist, excludes XML namespace URIs by name (they are
identifiers, never fetched), keeps `volume.html` / `command.html` /
`landing.html` at **zero** external origins, and asserts the literal string
`enabled: false` in `index.html`. Flipping the default breaks the build, not
just the claim.

**Attribution.** The Leaflet attribution control is on — NASA GIBS requires
it. Dark diagonal wedges over the ocean are orbital-pass gaps, and the note
above the map says so. Unexplained black on a risk map invites the worst
available reading.

---

## 6. Cross-cutting invariants that shipped

### 6.1 Every page fully vendored, except one opt-in origin

`web/vendor/` holds Leaflet CSS (14 KB), Leaflet JS (144 KB), `three.min.js`
(590 KB). No fonts, no CDN, no analytics, no framework. The volume view uses
three.js r128 exactly because r128 already ships `DataTexture3D`,
`RawShaderMaterial`, `GLSL3` and `NearestFilter` — no bundle upgrade needed.

### 6.2 One threshold, defined once

`REVIEW_THRESHOLD = 0.0909` comes from `quality/escalation.py`, which computes
it from the asymmetric cost `C_fa / (C_fa + C_miss)` at 10:1. The volume's
isosurface, the dashboard's tier bands and the review queue all read the same
value. It is not a free parameter — nothing anywhere in the UI hardcodes a
different one.

### 6.3 Three status tiers, never two

`AUTO_OK` / `REVIEW` / `REFUSE`, from `BustPrediction.review_tier`. A binary
render would collapse "refused" (elevated unknown risk, 23.4%) into "low risk"
(scored, 3.4%). Every visualisation carries the tier through to render — the
dashboard cards, the review queue chips, the volume shader.

### 6.4 The narrative narrows, and stays narrower

`FRONTEND_LOGIC.md` §8 lists what the UI may and may not claim. The landing
page is the only surface that states the ENS margin, and it states whatever
the registered tests say: since D-025, that the model outranks a real
50-member ensemble over the 2022 held-out season, +0.032 [+0.014, +0.048] on
120 init dates; since D-026, that outside 2022 this is not established and in
2019 the ensemble won. The intervals are on the page, not tucked below the fold. The
dashboard makes no skill claim at all. (An earlier version of this section
said the dashboard used the "comparable" wording too; it never did.)

---

## 7. Sub-project E — what did not ship, and why

Deployment and hardening — public hosting, an accessibility audit, performance
budgets, a Content Security Policy. Blocked on a hosting decision. AWS was
explicitly excluded ("no aws deployment, funds over"). GHCR + a free-tier host
is the zero-cost path if you want it as the default.

The `publish.yml` workflow has never fired — it triggers on push to `master`,
and the working branch is `local-llm-and-image-slimming`. Nothing has been
Docker-built for the registry, so the image size has never been measured.

---

## 8. Tests, mapped to what they protect

| test | what breaks it |
|---|---|
| `tests/test_web_pages.py` (21 tests) | dashboard `<title>`, viridis ramp present, hatch id, imagery default off, no CDNs, keyboard hint text |
| `tests/test_voxel_grid.py` | volume renders the model's own grid, not a resembling one |
| `test_volume_view_samples_nearest_on_every_axis` | `NearestFilter` on all three axes — a shader cannot assert on itself |
| `test_pick_and_picture_march_the_same_way` | both shaders built from one `PRELUDE`; traversal and opacity defined once |
| `test_north_is_minus_z_everywhere` | the mirror comes back |
| `test_outline_never_runs_through_a_cell_the_grid_does_not_cover` / `test_grid_coast_is_never_more_than_one_cell_from_the_outline` | the two geographies drift apart |
| `test_slice_is_a_whole_lead_day` | a float slice (a Day 3.5) |
| `test_pick_names_the_voxel_that_was_seen` | first-hit picking returns |
| `test_review_flag_comes_from_the_real_probability` | 8-bit *p* decides REVIEW again |
| `test_focus_and_slice_dim_opacity_never_colour` | a highlight whitens a value |
| `test_flythrough_hardcodes_no_case_number` | a scripted number in the flight |
| `test_column_view_uses_the_dashboard_ramp` / `…_is_seen_from_the_south` | green→red or upside-down India in the fallback |
| `test_readout_never_prints_a_probability_for_a_refused_cell` | refusal never reads as a number |
| CI `invariants` job | GenAI off by default, no write-capable tool on the model, numeric grounding refuses invented probabilities |
| CI `airgap` job | serving path opens no socket at import; no external origin outside the allowlist; `enabled: false` literal present |

---

## 9. Things this file will not lie about later

The list, so the next contributor can check them:

- Refused ≠ low-risk. The 6.9× number is why.
- Imagery is opt-in and dated to the lead, not the clock.
- The margin over a real ensemble is +0.0316 [+0.0141, +0.0485] on 120 init
  dates of **one** season, 2022, from a test registered before the data
  (D-025). It does not replicate: over 2019–2021 it is +0.0036 [−0.0059,
  +0.0127], and in 2019 the ensemble wins (D-026).
  Every surface that talks about skill gives the interval, and none stretches
  it beyond that season.
- No transformer, BERT, LSTM, RF or LightGBM is served. The served model is
  XGBoost + isotonic + TreeSHAP + Mahalanobis, and landing-page copy calls it
  that. An MLP exists in the codebase as a registered S3 candidate (D-027); it
  outranks XGBoost across 2019–2022 but is not served.
- The 3-D volume is blocky on purpose. Smooth filtering is a bug.
- India is not mirrored and not upside down in either 3-D view: north is `−z`
  and the camera starts on the south side.
- A full-volume pixel is a blend along the ray; only a slice shows a value.
- Nothing in the fly-through is scripted. It says what the data for that date
  says, including false alarms.

If any of the above stops being true, this file is wrong before the code is.
