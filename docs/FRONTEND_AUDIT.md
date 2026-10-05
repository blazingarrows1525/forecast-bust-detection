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
- system fonts only (no `@font-face`) and no `SIH`

## Next

Phase B (skill-driven design) is in progress: `PRODUCT.md` written through
Impeccable `init`; the direction round is open on Impeccable's decision page
(seed `61afe369`). The chosen world becomes the direction contract, then the
vertical slice: landing → real case → 2-D map → selected region/lead → 3-D →
volume → back, with F1–F4, F6, F12 and F13 fixed first.
