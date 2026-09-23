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
| **C** | Volumetric risk field | `web/volume.html` (761 lines) | shipped |
| **D** | Date-matched imagery | `web/index.html` (opt-in layer) | shipped |
| **E** | Deployment & hardening | — | blocked on hosting |

**169 tests pass**, all five CI checks green. The four pages ship with three
vendored files — Leaflet CSS/JS and `three.min.js` — and one 21 KB precomputed
grid. No fonts, no CDN, no analytics, no framework.

The volume view has [its own design doc](superpowers/specs/2026-09-20-volumetric-risk-field-design.md);
this file does not duplicate it.

---

## 2. Sub-project A — landing page (`web/landing.html`)

**Job:** tell the June 2022 story in a way a judge can read in under a minute,
without a single number that isn't live from the API.

**The invariant.** Every claim on the page reads live from `/api/convergence`.
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
is applied literally: the page says "comparable to a real 50-member ensemble
on the days we could afford", never "beats". The +0.025 interval that contains
zero is on the page, not below the fold.

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
21 KB, precomputed from `weights_129x135_*.parquet` — the same exact
polygon-cell overlap the feature pipeline itself uses, so the rendered volume
sits on precisely the grid the model was trained on rather than a second grid
that merely resembles it. `test_grid_is_the_model_s_own_grid` asserts it.

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
page and the dashboard both apply that list literally: "comparable to a real
50-member ensemble", never "beats it". The +0.025 [−0.008, +0.058] interval is
on the landing page, not tucked below the fold.

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
| `test_pick_and_picture_march_the_same_way` | pick shader and display shader share step count and axis swizzle |
| `test_readout_never_prints_a_probability_for_a_refused_cell` | refusal never reads as a number |
| CI `invariants` job | GenAI off by default, no write-capable tool on the model, numeric grounding refuses invented probabilities |
| CI `airgap` job | serving path opens no socket at import; no external origin outside the allowlist; `enabled: false` literal present |

---

## 9. Things this file will not lie about later

The list, so the next contributor can check them:

- Refused ≠ low-risk. The 6.9× number is why.
- Imagery is opt-in and dated to the lead, not the clock.
- The margin over a real ensemble is +0.025 with an interval that contains
  zero. Every surface that talks about skill says so.
- No transformer, BERT, LSTM, RF or LightGBM appears anywhere. It is XGBoost +
  isotonic + TreeSHAP + Mahalanobis. Landing-page copy calls it that.
- The 3-D volume is blocky on purpose. Smooth filtering is a bug.

If any of the above stops being true, this file is wrong before the code is.
