# Frontend audit — baseline, 2026-10-05

The state of the four product pages before the redesign, measured against
`FRONTEND_LOGIC.md` (the authority for claims and behaviour), `PRODUCT.md`,
and the tests in `tests/test_web_pages.py`. Evidence is the real running app
on `master` (`4d46424`) with the shipped store (`bulletins.sqlite`, 79,900
rows, store v0.2.0).

## Method and evidence

- **Server:** `python -m uvicorn fbd.api.app:app --app-dir src --port 8912`
  (now also `.claude/launch.json` entry `fbd`).
- **Captures:** `python scripts/capture_screenshots.py --label baseline` drives
  headless Chrome (SwiftShader WebGL) over the DevTools protocol at 1440×900,
  1280×800, 1024×768, 768×1024 and 390×844, with full-page captures at 1440
  and 390. That gives 20 viewport captures plus full pages and
  `report.json`, all in `artifacts/frontend/baseline/` (local, 2.9 MB, not
  committed). The same script captures "after" under identical conditions.
- **Interactive checks:** the in-app browser at 1440×900 for console errors,
  network requests and load behaviour.
- **Tests:** `PYTHONPATH=src pytest tests/ -q` → **330 passed**.

## Machine findings

| check | result |
|---|---|
| console errors | only `favicon.ico` 404 (every page, no favicon declared) |
| failed requests | none besides the favicon |
| horizontal overflow | **command.html at 390 px: document 650 px wide** |
| WebGL2 | available (SwiftShader); volume fails loudly without it (by design) |
| external origins | none (test-enforced); imagery opt-in only |
| requests per page | landing 4, index 9, command 5, volume 5 |

## Scores (1–5) and findings

Dimensions: purpose clarity, hierarchy, typography, colour/contrast, layout,
operational density, data legibility, geographic correctness, motion purpose,
discoverability, responsive, accessibility, empty/error states, scientific
honesty, performance, cross-view consistency, distinctiveness.

| dimension | landing | index (2-D) | command (3-D) | volume |
|---|---|---|---|---|
| purpose clarity | 4 | 3 | 3 | 4 |
| hierarchy | 3 | 2 | 3 | 4 |
| typography | 3 | 3 | 3 | 3 |
| colour & contrast | 3 | **2** | 3 | 3 |
| layout | 3 | 3 | 3 | 4 |
| operational density | — | 3 | 3 | 3 |
| data legibility | 4 | **2** | 2 | 3 |
| geographic correctness | — | 5 | 4 | 5 |
| motion purpose | 4 | 4 | 4 | 4 |
| discoverability | 3 | 3 | 3 | 4 |
| responsive | 3 | **1** | **1** | 3 |
| accessibility | 3 | 3 | 2 | 3 |
| empty/error states | 3 | 3 | 3 | 4 |
| scientific honesty | 5 | 4 | 4 | 5 |
| performance | 5 | 5 | 4 | 3 |
| cross-view consistency | 3 | 3 | 3 | 3 |
| distinctiveness | **2** | **2** | 3 | 4 |

### What is strong and must be preserved

- **Scientific honesty is excellent.** The landing page derives its claims
  from `/api/convergence` and `/api/metrics`; refused leads break the line and
  are hatched; ENS verdicts, the backtest and the combination are all
  API-gated. These are pinned by tests and stay exactly as they are.
- **The risk ramp** is viridis, colour-vision tested (`--c0…--c4`), and shared
  between the 2-D and column views. Refusal is a pattern, never a hue.
- **The volume view** is the most considered page. It has nearest-sampled
  voxels, honest legend copy ("Outline smoothed for orientation…"), keyboard
  slice controls, a fly-through and a WebGL2 failure path.
- **Keyboard shortcuts and help overlay** on the dashboard; theme toggle.
- **Every page loads with no console errors** apart from the favicon.

### Findings below 4

Severity: **P0** breaks a task or a rule; **P1** materially weakens a task;
**P2** polish.

| # | page | finding | evidence | severity | consequence | fix | effort | validation |
|---|---|---|---|---|---|---|---|---|
| F1 | index | At 390 px the sidebar covers the map; about 20 px of map is visible | `index@390x844.png` | **P0** | The map, the core view, is unusable on a phone | Stack map above a scrollable panel on narrow widths | M | 390/768 captures |
| F2 | command | At 390 px the toolbar and sidebar overflow; the document is 650 px wide | `report.json` `scrollWidth: 650` | **P0** | Horizontal scroll; controls off-screen | Responsive toolbar + stacked panel | M | overflowX false at 390 |
| F3 | index | Review-queue probability pills are viridis-dark fills with dark text (e.g. "2.4%" on `#3b0f70`) | `index@1440x900.png` | **P0** | Values unreadable; contrast far below 4.5:1 | Text colour chosen per fill luminance, or a separate bar + numeral | S | contrast check |
| F4 | index | Default date is the middle of the list (2022-06-01): every region is `<5%`, so the first view is a uniform purple map | `index@1440x900.png` | P1 | The first screen demonstrates nothing; the forecaster sees no reason to look | Open on the URL's date if given, else the most recent date with REVIEW-tier rows | S | first-load capture |
| F5 | index, command | Lowest ramp stop (`#440154`-like) is close to the near-black ground, so low-risk regions merge with the background and boundaries vanish | 1440 captures | P1 | Geography is hard to read where nothing is happening, which is most of the map | Change the ground and surfaces, not the tested ramp | M | captures + CVD test still green |
| F6 | landing | Primary action is below the fold at 1440×900 (CTA at y≈1350) | `landing@1440x900-full.png` | P1 | Visitors must scroll to act | Action inside the first viewport | S | 1440/1280 captures |
| F7 | landing | Equal-card rows twice ("three conditions", "three ways in") plus a four-up stat row | full capture | P1 | Template rhythm; reads like any SaaS page; hurts distinctiveness | Recompose sections around the product's own forms | M | critique |
| F8 | all | The visual world is a generic dark dashboard (GitHub-like slate, single blue accent) | all captures | P1 | Nothing says *this* product; the brief asks for a distinctive identity | Replacement visual world (Impeccable new-work) | L | finish review |
| F9 | index | Toolbar has nine equal-weight buttons; primary controls (date, lead) do not lead | 1440 capture | P1 | Slow to find the controls that matter | Hierarchy: date/lead first, view toggles grouped, links demoted | S | critique |
| F10 | command | Tall low-risk columns are near-black and hide the map they stand on; no lead-axis labels in the scene | `command@1440x900.png` | P1 | Spatial reading suffers; "which column is which" is hard | Thinner graduated staffs, lead marks in-scene, lighter ground | M | captures |
| F11 | index | Legend sits below the fold of the sidebar at 1440×900 | 1440 capture | P2 | Colour key is not visible while reading the map | Legend docked with the map | S | capture |
| F12 | all | No cross-page context: opening the 3-D view from the map loses the selected date/lead/region | code: links have no query | P1 | The journey breaks at every page change | URL query `?date=&lead=&region=` read and written by every page | M | navigation test |
| F13 | all | No favicon (404 on every load) | `report.json` | P2 | Console noise; tab has no identity | Inline SVG data-URI favicon (not an external origin) | S | report.json clean |
| F14 | index, command | Selected-region panel shows a bare em-dash placeholder before selection | captures | P2 | Reads as broken rather than "select a region" | Real empty state with instruction | S | capture |
| F15 | volume | 8 fps under software rendering; on GPU hardware it is fine | `volume@1440x900.png` footer | P2 | Slow on machines without a GPU | Leave raymarch; add a resolution-scale note/control only if measured on real hardware | — | measure on GPU |
| F16 | command | No keyboard alternative for selecting a column; picking is mouse-only | code: no key handlers | P1 | Keyboard users cannot reach column details | Queue rows focusable and synced to the scene | M | keyboard test |

## What must not change (test-pinned)

The redesign keeps, verbatim where a test reads it:
- the `--c0…--c4` ramp in `:root{` of `index.html` and `command.html`
- `OOD_FILL = "url(#oodHatch)"` and `<pattern`
- `refusalTexture()` and the south-facing camera `theta: Math.PI/2`
- the landing page's derived claim (`r.bust_probability > r.baseline_probability`),
  segmented lines (`segments.push(run)`), `url(#hatch)`,
  `getJSON("/api/metrics")`, every ENS-verdict wording, the backtest
  statement, combination gating and `prefers-reduced-motion`
- keyboard handlers and `id="help"`
- the override reason rules and dismiss-a-refusal warning
- imagery opt-in, attribution and degradation
- system fonts only (no `@font-face`) and no competition branding

## Next (historical)

Done on 2026-10-05: the redesign shipped (PR #7) with F1–F14 and F16 fixed;
see `docs/FRONTEND_IMPLEMENTATION_REPORT.md`.

---

## Re-audit, 2026-10-06 (after the redesign)

A second pass on the shipped four-view frontend, run under the revised
master prompt. It used two installed skills:
- Taste `redesign-existing-projects` (scan, diagnose, fix)
- Impeccable `audit` (technical: accessibility, performance, theming,
  responsive, implementation integrity)

The evidence is today's `master` (`dadac94`), served from an isolated
worktree on port 8913, with captures in
`artifacts/frontend/reaudit-baseline/` (20 captures: 0 overflow, 0 console
errors, 0 failed requests). Tests: 346 passed, 1 skipped (the skip needs
the gitignored district shapefile, which the worktree does not have).

### Audit health score (Impeccable `audit`)

| # | dimension | score | key finding |
|---|---|---|---|
| 1 | Accessibility | 3 | No skip link; the map and columns views have no `<main>` and no `h1`; map regions are reachable by keyboard only through the top-12 queue |
| 2 | Performance | 2 | The columns view leaks 340 geometries and materials, plus a canvas texture, on every lead or view change, and renders 60 fps while idle |
| 3 | Responsive | 3 | No overflow at any of 5 viewports, but the 3-D canvas has no `touch-action`, no pinch zoom and no `pointercancel` handling; controls are 30 px on touch screens |
| 4 | Theming | 3 | Full token system, both themes AA (lowest text pair 4.84:1). Hard-coded help-overlay colour and `z-index: 9999`. |
| 5 | Implementation integrity | 4 | The detector reports only the refusal hatch (intentional, test-pinned) |
| | **total** | **15/20** | **Good**: address performance and touch |

**Implementation integrity verdict: pass.** One product-specific system
(chart paper, slate shell, gauge staffs), with every number from the API.

### Contrast (measured, WCAG 2.x)

| theme | lowest text pairs |
|---|---|
| dark | `ink-3` on `plate` 5.21; on `ground-2` 5.59 |
| light | `base` (ochre) on `ground-2` 4.84; `ink-3` on `ground-2` 4.90 |

Accent buttons are 7.5:1 (dark) and 10.0:1 (light). The lowest viridis
stop on the map ground is 1.04:1 as a fill. Regions are told apart by their
`--map-edge` outlines, not by the fill.

### Findings and fixes

| # | sev | where | finding | fix |
|---|---|---|---|---|
| R1 | P1 | all pages | no skip-to-content link | a skip link to `#main` on every page |
| R2 | P1 | index, command | no `<main>` landmark and no `h1`; the heading outline starts at `h2` | `<main id="main">` around the work area; a visually hidden `h1` naming the view |
| R3 | P1 | command | GPU leak: `buildColumns()` creates 340 `BoxGeometry` + `MeshLambertMaterial` per rebuild and never disposes them; `buildStaff()` leaks a `CanvasTexture` each time | Two shared geometries, a material cache keyed by colour and opacity, and dispose the staff texture |
| R4 | P1 | command | 3-D touch: no `touch-action: none`, no pinch zoom, a drag left stuck on `pointercancel` | pointer map with pinch-to-zoom; clear state on cancel or lost capture |
| R5 | P1 | index | Only 12 queue rows are keyboard-reachable; the help text promised "/ jumps to a subdivision" and the key went to the date | A **"Jump to subdivision"** select listing all 34 regions; `/` focuses it |
| R6 | P2 | command | Renders every frame while idle | render on demand (a dirty flag set by orbit, rebuild and resize) |
| R7 | P2 | command | No hover readout; selection is not marked in the scene; picking from the queue does not bring the column into view | Hover tooltip (region, lead, value or "refused"); outline the selected column; ease the camera target to it (instant under reduced motion) |
| R8 | P2 | index | On a phone, choosing a queue row can select a region that is off-screen | pan the map to the region when it is not in view |
| R9 | P2 | all | Touch targets 30 px | 44 px minimum under `pointer: coarse` |
| R10 | P2 | site | An unknown URL returns raw JSON `{"detail":"Not Found"}`, a dead end | `web/404.html` (StaticFiles serves it automatically), offline, with the way back |
| R11 | P2 | fbi.css | A global reduced-motion kill on `*` | scope it to the elements that move |
| R12 | P3 | index | help overlay colour hard-coded; `z-index: 9999`; Leaflet credit carries a flag emoji | tokens, a small z-scale, plain "Leaflet" prefix |
| R13 | P3 | index, command, volume | no meta description; caret colour unthemed | add both |

**Landing: craft opportunities carried from the finish review** (not
defects):
- **L1:** the onset-chart *geography* is absent. Add a chart-paper locator
  map of the case region drawn from `/api/regions`.
- **L2:** the evidence register prints its intervals but does not draw
  them. Set each reading on a scale with its interval band and the
  baseline marked.

**Unchanged by design:**
- **F15** (volume frame rate under software rendering) needs a GPU
  measurement.
- **The volume view's render loop and shaders** are test-pinned; this pass
  does not touch them.
