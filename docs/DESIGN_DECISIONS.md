# Design decisions: the Forecast Bust Intelligence world

Why the frontend looks the way it does, and which rule won each time two
rules disagreed. `DESIGN.md` describes the system as built; this file records
the reasoning. Written 2026-10-05 with the redesign on branch
`frontend-redesign`.

## 1. One world, three materials

The direction round (Impeccable `concept-seed --scope direction --mode
persuade`, seed `61afe369`) dealt **River Gauge Staff**. The alternatives
offered were Impeccable's pick, **Monsoon Onset Chart**, and the standing
exit, **the category standard**. You chose "all 3 of them combined".

A median blend of three worlds gives you none of them. So each world got one
job instead:

| material | world it comes from | job | where |
|---|---|---|---|
| chart paper: cool off-white `#f2f4f0`, blue-black ink `#172133`, hairline graticules | Monsoon Onset Chart | the ground the case is explained on | landing |
| wet-concrete slate: `#15191c` ground, `#20262b` plates | category standard | the calm shell for working under time pressure | map, columns, volume |
| gauge enamel: white plates, E-graduations in gauge indigo `#27407a` (`#8fa6e8` on slate) | River Gauge Staff | **the reading**, the one signature | landing staffs, the selected-reading staff on the map, the 3-D lead staff |

### The signature move: taking a reading

A forecast is read like a river gauge.

- **Landing:** one enamel staff per forecast. The served bust probability is
  the water level on that staff, the ink line is its meniscus, and the review
  level (0.0909, the cost-optimal threshold) is a band painted across every
  staff.
- **Map:** the selected region's panel draws the same reading as a horizontal
  staff.
- **Columns (3-D):** a gauge staff stands off the west coast and reads D01 to
  D10 up the lead-day axis, with the highlighted lead marked.

### Two translations, named

- **No danger paint.** Real gauge staffs carry red and yellow danger bands.
  This product issues no warnings, and the risk ramp is the colour-vision-tested
  viridis. So the staffs carry a *review* level in ink, and the copy never says
  "danger level".
- **Bahnschrift for the stencilled lettering.** The world wants stencilled
  signage numerals. A webfont is banned (offline guarantee, and
  `tests/test_web_pages.py` rejects `@font-face`). Bahnschrift is Windows'
  DIN signage face. The fallbacks are "DIN Alternate", "DIN 2014",
  "Roboto Condensed", "Arial Narrow", then system-ui. Off Windows, the display
  voice falls back to the nearest available cousin; this is accepted.

## 2. When the skills disagreed

| conflict | rule A | rule B | decision and why |
|---|---|---|---|
| stack | Taste defaults to Next.js, Tailwind, Motion, `next/font` | the product has no build step and must run air-gapped | Native CSS, inline SVG and the vendored Three.js r128. The no-build, offline core is validated and test-enforced; a framework would break both. |
| display font | Impeccable's craft floor wants a self-hosted display face | tests ban `@font-face`; offline | Bahnschrift system font (see §1). |
| icons | Taste: "never hand-roll SVG icons", use an icon library | Impeccable floor: no glyph icons (▶ ◐ ‹ ›) | Text labels ("Scrub", "Stop", "Theme", "Fly-through"). The only drawn marks are two 10-px chevron steppers and the ruled time arrow on the landing figure. An icon library would add a dependency for three marks. |
| themes | Taste: support both themes | the direction contract: the landing's ground is chart paper | The landing is chart paper in every OS colour scheme (finish review, GROUND). The working views are dark by default with a light (chart-paper) theme on the `Theme` toggle. |
| eyebrows | the old landing had uppercase kickers on every section | both skills ban eyebrows and kickers | Removed. Headings carry the section. |
| numbering | the old page numbered the three bust conditions and the three views | the craft floor bans decorative numbering | Removed. The conditions are a conjunction joined by "and"; the views are an unordered route. |
| em-dashes | the old copy used them freely | Taste: zero em-dashes in visible text | Rewritten with colons, commas and full stops. The null placeholder "—" became "n/a". En-dashes remain only in numeric ranges ("5–10%", "2016–2022"), which is typographic use, not a separator. |
| refusal hatch | the detector flags `repeating-stripes-gradient` as decoration | `FRONTEND_LOGIC.md` §2.1 and the tests: refusal is a pattern, never a hue | **Kept on purpose.** It is the most important encoding in the product, not decoration. This is the only detector finding left. |
| second hue | the contract: "gauge indigo, the single accent" | the baseline needs to be told apart from the served line at a glance | The baseline keeps ochre (`--base`) as a *data-series* colour, not an accent. It is also encoded by shape (dash and notch), so it never depends on hue alone. Outcomes ("bust") moved from orange to ink weight. `--outcome` survives only as the fill in the map's Observed view, where a fill needs a colour. |

## 3. What did not change, and why

- **The risk ramp** `--c0…--c4` (viridis) sits first in each page's own
  `:root{` block. The colour-vision test reads it from there, and
  `command.html` must match `index.html`.
- **Refusal** is a hatch everywhere: the SVG pattern on the map, the canvas
  texture on the columns, the swatch, and the ghost plates on the landing.
- **The camera** in the columns view looks from the south
  (`theta: Math.PI/2`).
- **Volume shaders**: untouched. That view got the shared shell, nav, favicon
  and URL date, and nothing else.
- **Every claim** on the landing is still derived at load from
  `/api/convergence` and `/api/metrics`, word for word in logic.
- **The override flow**: reason required and never prefilled, plus the
  dismiss-a-refusal warning.

## 4. Fixes the redesign made that are not visual

- **Context travels in the URL** (`web/fbi.js`). `?date=&lead=&region=` is
  validated (date pattern, lead 1–10, an uppercase region id), written with
  `replaceState`, and carried by every `data-carry` link. A reading opened on
  one view is the reading opened on the next.
- **Stored XSS closed.** The override log used to insert typed reasons and
  names into the page as HTML. Everything typed now passes through `esc()`.
  The reasons from the API are escaped too.
- **The case opens by default.** The map and the columns open on 2022-06-14
  at D04. That is the Assam case, and the reading with the widest gap.
  Previously the map opened on a flat mid-season date. On the landing, the
  primary action's link is recomputed from the data to point at the widest
  gap.

## 5. Review record

- **Finish review, round 1** (`impeccable-finish-reviewer`): disposition
  `fix`, with eight material fixes:
  1. the landing ground
  2. the painted review band
  3. enamel and water
  4. glyph icons
  5. decorative numbering
  6. palette
  7. legend units
  8. the column view on phones

  All eight were addressed; the verdict pass is recorded in
  `docs/FRONTEND_IMPLEMENTATION_REPORT.md`.
- The roll record and both rounds are also kept in
  `.impeccable/surfaces/web-landing-html.md`.

## 6. Re-audit trade-offs (2026-10-06)

The findings and evidence are in `docs/FRONTEND_AUDIT.md`, under "Re-audit".
These are the choices a reviewer could question.

- **A jump list, not keyboard-focusable map polygons.** 34 tab stops on a
  Leaflet map would bury the queue and the controls. One native `<select>`
  reaches every subdivision, works with any screen reader, and `/` opens
  it. The map still answers pointer clicks.
- **Pan, never zoom, to an off-screen pick.** The forecaster chose the zoom.
  A queue pick that changed it would lose the national view they were
  reading. Under reduced motion the pan is instant.
- **Render on demand in the columns view.** Idle frames cost a laptop GPU
  and battery for nothing. The trade is a camera-signature comparison every
  animation frame, which is cheap.
- **The locator map is drawn, not tiled.** The landing stays offline. The
  map is the project's own subdivision GeoJSON, decimated to about 1 px
  (609 KB in, about 106 KB of SVG out). It loads only when the reader
  scrolls near it, so the first paint does not wait for it.
- **Scales on the register use the reading's own range.** AUROC runs 0.5
  to 1 (chance to perfect). BSS and ECE use rounded maxima of their
  intervals. A shared 0–1 axis would flatten every interval into a dot.
- **No new hues.** The interval band is the water tone, the baseline keeps
  its ochre notch, and refusal keeps the hatch. The 404 page is the same
  chart paper, with a hatched staff that has no reading on it.

No external references were consulted for this pass, so there is no
source ledger.
