# Immersive frontend: handoff report

2026-10-18 · branch `immersive-frontend` · decision D-038 · surface brief
`.impeccable/surfaces/web-landing-html.md`.

Labels: **IMPLEMENTED** (built and in the branch), **TESTED** (a test or a
measured check confirms it), **UNVERIFIED** (built, not confirmed on the
target), **PROPOSED** (not built).

## A. Baseline audit

**Pages:** landing (`landing.html`), map with review queue (`index.html`),
command centre (`command.html`, 3-D columns), risk volume (`volume.html`). The
date, lead day and subdivision travel in the URL between them (`fbi.js`).
Stack: static HTML, CSS and vanilla JS, vendored three.js r128 and Leaflet,
FastAPI serving precomputed SQLite. No build step, no external origins.

**Baseline, recorded before any edit:**
- `PYTHONPATH=src python -m pytest tests/ -q`: 449 passed, 1 skipped.
- `scripts/capture_screenshots.py --label baseline`: 20 captures, no console
  errors, exceptions, failed requests or overflow.
- Command centre: 344 draw calls, 18,381 triangles, 1.31 ms per frame (the
  owner's RTX 3060).

**Strengths:**
- every number on every page comes from the API
- refusal is a hatch, never a hue
- tests pin the data contracts and wording rules
- render-on-demand in the command centre

**Confirmed defects:**
- The landing was a light page with no spatial scene. The brief asked for
  the opposite.
- The command centre had no playback, no hover outline and no WebGL fallback.
- The volume raymarched every frame while idle.
- No view showed a reading's provenance in one place.
- `fbi.css` let a control's `display` override `[hidden]`, so a hidden button
  could stay visible.

**Protected invariants:**
- no hardcoded case numbers, and every claim derived from the rows
- the model line breaks at a refusal
- refusal as a hatch
- the CVD-safe viridis ramp, identical across pages
- nearest sampling in the volume
- imagery stays opt-in and date-matched
- the override reason is required
- no external origins and no `@font-face`
- no competition branding

## B. Design direction

**Art direction:**
- **The roll:** Impeccable assigned a synoptic station-plot chart (seed
  `1601c50a`), fused with the brief's pinned material: a dark, locally
  rendered globe with India in focus.
- **The world:** a night orbit. The void is graphite-indigo; the globe is a
  plotting sphere with a 10-degree graticule and a cool limb.
- **India:** raised on the model's own 0.25-degree grid, one column per cell.
- **Station plots:** in cool-white ink. Upper left is the served reading,
  lower left the baseline, upper right the lead, lower right the observed
  outcome, and the subdivision beneath.

**Type and colour:**
- **Type:** the system signage face (Bahnschrift) for numerals and headings,
  system UI for prose.
- **Colour:**
  - one accent, gauge periwinkle
  - risk only on the five viridis stops
  - the ensemble baseline in ochre, always dashed

**Controls (machined plates):**
- a top highlight and a darker lower edge
- a 1 px lift on hover
- an inset press
- pressed toggles that stay sunk with the accent underline
- flat when disabled
- a running hairline when busy
- segmented bars never lift a key
- no glass, blur or glow

**Motion:** one authored intro of about 4.6 s, in named states:
1. orbit
2. outline
3. boundaries
4. plotted

It is skippable by button, key, wheel or click, and absent under reduced
motion. Lead changes swap values without tweening.

**Why it fits:**
- A forecaster's world is synoptic charts and plotted stations. The globe
  orients the visitor; the plots carry the evidence.
- The tactile controls read as instruments, not marketing.

## C. Changes implemented (IMPLEMENTED)

**`web/landing.html` + new `web/orbit.js`: the hero scene.**
- **Grid and data:** a WebGL scene built from `/api/voxel-grid` and
  `/api/risk-cube`.
- **Which reading:** the one the case itself selects from `/api/convergence`,
  the widest served-over-baseline gap among leads that busted.
- **Interaction:**
  - station plots placed in the void with leaders
  - a hover tooltip after 140 ms
  - drag to turn, with inertia
  - click to open the reading in the command centre
  - a lead rail with Play and arrow keys
- **A screen-reader table** of all readings for the plotted lead.

**`web/landing.html`: the rest of the page.**
- **Quality tiers:** high, balanced, low and 2d; `?quality=` overrides.
- **2-D fallback:** a canvas drawing of the same cells.
- **Rendering:** no frames off screen or while the tab is hidden; resources
  are disposed on `pagehide`.
- **The case:** now a meteogram of plotted stations, with a key explaining
  the station convention.
- **Unchanged:** all API-driven claims and wording rules.

**`web/fbi.css`:**
- the orbit material
- machined-plate control states
- `[hidden]` that holds
- the shared "Why this number?" style

**`web/fbi.js`:** `FBI.provenance(row, served)`. It shows:
- source, issue, valid date and lead
- the store version and build date
- the base model
- how the served number and the reasons are made
- distance from training against the refusal threshold, and what that check
  covers
- review tier and data quality
- "unavailable" for anything missing

**`web/command.html`:**
- lead-day playback (step, play, reverse, speed, reset) showing the lead and
  its valid date
- a hovered-cell outline and a short tooltip delay
- a damped focus on a clicked column
- "Focus the top case"
- a WebGL check that sends the reading to the map
- a low-quality tier
- provenance

**`web/index.html`:** provenance in the selected reading, re-rendered once the
served metadata arrives.

**`web/volume.html`:** idles without input; any input, flight, focus easing or
camera move wakes it.

**`src/fbd/api/app.py`:** `/api/metrics` `served` adds `ood_threshold` and
`generated_at` from the store's meta. A malformed value is null.

**Data-backed vs illustrative:** everything shown is data-backed. Nothing is
illustrative; no physics layer was built.

## D. Verification

**Tests (TESTED):**
- `PYTHONPATH=src python -m pytest tests/ -q`: 451 passed, 1 skipped. That is
  the baseline plus two new API tests.
- `scripts/capture_screenshots.py --label after3` with `check_captures.py`:
  20 captures (4 routes × 1440, 1280, 1024, 768, 390 px), 0 problems.

**Extra captures** (TESTED, headless Chrome with SwiftShader):
- 1366×650
- the `?quality=2d` fallback
- reduced motion
- no console errors in any of them

**Performance** (TESTED):

| scene | draw calls | triangles | resources after 30 lead changes |
|---|---|---|---|
| landing | 6 | about 110,000 (5,107 cells) | 6 geometries, 0 textures |
| command centre | 344 | 18,381 | 6 geometries, 340 meshes |

- **Command centre at 1.31 ms per frame:** instancing was not justified, so
  not done.
- **Volume idling:** reports "idle, not redrawing" about 1.5 s after the last
  input and wakes on a pointer move.

**Accessibility** (TESTED where a check exists):
- keyboard: the lead rail takes arrow keys, the station plots are links, and
  the playback controls are buttons
- visible focus rings
- reduced motion removes the intro and the transitions
- the hero's readings are mirrored in a table for screen readers
- refusal is drawn as a pattern and named in words

**Offline** (TESTED): no external origins; the CI allowlist check is unchanged.

**Interactions** (TESTED in the browser through state checks):
- step, reverse playback, reset and "Focus the top case" in the command
  centre
- provenance on the map and in the command centre
- lead changes on the landing, with the call to action following the plotted
  lead

**UNVERIFIED:**
- frame rate on real hardware for the landing and volume (headless captures
  use software WebGL)
- touch behaviour on a physical phone
- screen-reader output in a real screen reader

**Design review:** two Impeccable finish-review rounds.
- **Round 1 (fix):**
  - the hero lit a forecast that held
  - India was too small
  - plots were boxed and over land
  - the ground drifted to blue-black
  - layout details
- **Round 2 (fix):** six fixes resolved, two recorded as adaptations (the
  title block's position and the display face), and two small regressions
  (a leader crossing India at 1280, the limb behind the headline at 1440).
  Both were fixed and checked in captures afterwards, but not re-reviewed.

**Deployment:** not checked here. The published image rebuilds on merge to
master; Render needs a manual redeploy.

## E. Remaining work

**Required before a demo:** none known. Open the landing once on the demo
machine to confirm its frame rate.

**Recommended next:**
- **Display face:** decide whether to self-host one. That would relax the
  system-fonts rule (owner decision).
- **Chart language below the hero:** carry the chart language into the long
  read, as graticule rules or a chart margin. The review called the sections
  below the hero a plain dark document.

**Optional experiments (PROPOSED):**
- an illustrative atmosphere/Coriolis layer, clearly labelled and off by
  default
- a WebGL picker on the landing for keyboard users (today the station plots
  and the table cover it)

## F. Career learning notes

**WebGL scene management (`web/orbit.js`).**
- One `THREE.InstancedMesh` draws 5,107 cells in one call. Per-cell data
  rides in `InstancedBufferAttribute`s (colour, refusal flag, region index,
  plot delay).
- The rise animation is a shader uniform, so the CPU never rebuilds geometry.
- Hover writes a uniform; picking uses `Raycaster.intersectObject(instanced)`
  for an `instanceId`.
- Render on demand: request a frame only when something changes.

**State synchronisation.**
- The URL is the shared state (`FBI.read`/`FBI.write` in `web/fbi.js`). Every
  link marked `data-carry` is rewritten, so a reading opened in one view is
  the one opened in the next.

**Data visualisation honesty.**
- The meteogram breaks its line at a refusal (`segments.push(run)`).
- The 3-D columns never tween between lead days; colour bands come from one
  function shared by every view.

**Performance profiling.**
- `renderer.info.render.calls` and `.triangles`, and `memory.geometries`
  before and after 30 lead changes, catch leaks.
- Time 60 forced renders with `gl.finish()` for a per-frame cost (see the
  command-centre measurement above).

**Accessibility.**
- `prefers-reduced-motion` gates every animation.
- Entrance animations fail safe: the hidden state is a class that a timer
  removes.
- `[hidden]` must beat component `display` rules.

**Tests and gates.**
- `tests/test_web_pages.py` pins the data contracts as text.
- `scripts/check_captures.py` fails CI on overflow, console errors or failed
  requests.
- A headless Chrome over the DevTools protocol (`scripts/capture_screenshots.py`)
  gives repeatable captures without a new dependency.

**Deployment.** `.github/workflows/publish.yml` builds the image on master,
smoke-tests it read-only, and pushes `:latest` to GHCR; Render runs that image.
