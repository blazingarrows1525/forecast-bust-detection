# Frontend implementation report: the vertical slice

Branch `frontend-redesign` · 2026-10-05 · the Phase C and E deliverable of
the master prompt. The evidence comes from the running app with the shipped
store (`bulletins.sqlite` v0.2.0, 79,900 rows).

Companion files:
- `docs/FRONTEND_AUDIT.md`: the baseline and findings F1 to F16
- `docs/DESIGN_DECISIONS.md`: why the frontend looks this way
- `DESIGN.md`: the system as built

## 1. What was built

The journey this slice covers, with every step deep-linkable:

```
landing (the Assam case on ten gauge staffs)
  → "Open the Assam case"   /?date=2022-06-14&lead=4&region=ASSAM_MEGHALAYA
  → map + review queue, the reading selected (58.1%, D04, valid 2022-06-17)
  → Columns (3-D)           /command.html?date=…&lead=…&region=…
  → Volume                  /volume.html?date=…&lead=…&region=…
  → back to the Case
```

| file | change |
|---|---|
| `web/fbi.css` (new) | Shared world tokens: chart paper, the slate ops shell, the indigo accent, `--water`, the hatch. Also the Bahnschrift sign stack, the 3-px radius system, focus and selection styles, the top bar, controls, `kbd` and reduced motion. |
| `web/fbi.js` (new) | URL state. Reads and validates `?date=&lead=&region=`, writes it with `replaceState`, and rewrites every `data-carry` link. |
| `web/landing.html` | Rebuilt as the gauge-staff landing: a hero that fits the first viewport, ten enamel staffs, a case table, the bust definition, refusal, the evidence register and the route. Every claim function is kept in logic. |
| `web/index.html` | New shell and hierarchy, with the case as the default reading. The queue is now a readable register, the legend docks on the map, the mobile layout stacks, and the selected-reading panel gets a staff. Override text is escaped. |
| `web/command.html` | Lighter ground, thinner columns and faint cells for the lowest band. Adds an in-scene D01 to D10 staff, keyboard rows, URL state and a stacked mobile layout with the camera fitted to the screen. |
| `web/volume.html` | Light touch: the shared shell, nav, favicon, URL date, text labels and no em-dashes. **The shaders were not touched.** |

## 2. Audit findings: status

Status words follow the master prompt. "Implemented & verified" means the
captures, tests or a browser check show it.

| # | finding | status | evidence |
|---|---|---|---|
| F1 | map unusable at 390 px | **implemented & verified** | Map on top at 62vh, panel below, fitted to India's bounds; `index@390x844.png` |
| F2 | column view overflows at 390 px (650-px document) | **implemented & verified** | `scrollWidth` 650 → 375. The canvas is sized by CSS and the camera distance is fitted to aspect. |
| F3 | unreadable queue pills | **implemented & verified** | Value in ink beside a ramp chip; `index@1440x900.png` |
| F4 | first load on a flat mid-season date | **implemented & verified** | Opens on 2022-06-14 D04 unless the URL says otherwise; north-east lit |
| F5 | lowest ramp stop merges with the ground | **implemented & verified** | Light subdivision edges on the map; faint lowest band in 3-D; ramp unchanged and the CVD test is green |
| F6 | landing CTA below the fold | **implemented & verified** | CTA inside 1440×900, 1280×800 and 390×844 |
| F7 | equal-card template rhythm | **implemented & verified** | No cards anywhere: a register, a conjunction list and a route list |
| F8 | generic dark dashboard identity | **implemented & verified** | Combined-world direction contract; two finish-review rounds (§4) |
| F9 | nine equal-weight toolbar buttons | **implemented & verified** | Issue date and lead come first; view segment; tools; nav |
| F10 | dark columns hide the map; no lead labels in-scene | **implemented & verified** (desktop) | Faint lowest band, lighter land, west-edge D01 to D10 staff. On phones the staff is hidden and the axis is stated in words. |
| F11 | legend below the sidebar fold | **implemented & verified** | Key docked on the map; follows the Served, Baseline or Observed view |
| F12 | no cross-page context | **implemented & verified** | Browser check: volume links carried `?date=2022-06-14&lead=4&region=ASSAM_MEGHALAYA` |
| F13 | no favicon | **implemented & verified** | Inline data-URI SVG; capture report shows 0 failed requests (baseline: 2) |
| F14 | bare dash placeholders before selection | **implemented & verified** | Written empty states on the map panel, the column panel, the lead profile and the queue |
| F15 | volume at about 8 fps under software rendering | **not addressed** | Needs measurement on GPU hardware. The audit said to leave the raymarch alone. |
| F16 | 3-D picking was mouse-only | **implemented & verified** | Queue and profile rows take Tab, then Enter or Space. ←/→ change lead, ↑/↓ date, `r` resets, Esc clears. |

Fixed beyond the audit:
- **Stored XSS in the override log:** typed reasons are now escaped.
- **Baseline dashes:** the draw animation had been overriding the dash
  pattern, so the baseline drew solid. It now keeps its dashes.
- **Help text:** it promised "/ jumps to a subdivision", but the key
  focuses the date. The text now says so.

## 3. Verification

| check | result |
|---|---|
| `PYTHONPATH=src pytest tests/ -q` | **330 passed** (same count as baseline; every test-pinned literal kept) |
| `tests/test_web_pages.py` + `tests/test_voxel_grid.py` | 90 passed after every edit round |
| `python scripts/capture_screenshots.py --label after` | 20 captures across 5 viewports × 4 pages; **0 overflow** (baseline: 1), **0 console errors, failed requests or exceptions** (baseline: 2) |
| `impeccable detect --json` on the 5 changed files | 1st run: 10 findings. Now: 4, all `repeating-stripes-gradient` = the refusal hatch, kept on purpose (`docs/DESIGN_DECISIONS.md` §2) |
| em-dashes in visible text | 0 on all four pages (en-dashes only in numeric ranges) |
| external origins | none added; the CI allowlist scan still covers the pages (the favicon is a data URI) |
| browser check (in-app, 1440 and 375) | deep links restore date, lead and region; the selected reading, queue row and map outline are marked; no horizontal scroll |

Captures: `artifacts/frontend/after/` (local, not committed, same as the
baseline). Review copies are in `.impeccable/review/`.

## 4. Finish review (Impeccable)

| round | disposition | what it found |
|---|---|---|
| review | `fix` | Eight material fixes: the landing ground was swapped to slate under OS dark mode; the review level was one chart line rather than painted per staff; the plates did not read as enamel; glyph icons; decorative numbering; a second hue and a grey; legend units split; the column view on phones |
| verdict 1 | `fix` | 7 of 8 resolved. Phone staff partial. One regression: a reading label against a full-strength graduation. |
| verdict 2 | **`ship`** | Both remaining items resolved: the phone staff is hidden with the axis given in words, and reading labels sit on their own enamel plates. No regressions. Its ordering note (`NARROW` declared after `resize()`) was fixed anyway. |

"Ship" covers the scored fixes, not a fresh review of the whole surface. The
reviewer's ceiling notes are still open as craft opportunities, not defects:
- the onset-chart geography is absent from the landing (no map of the case
  region, no isochrones)
- the evidence register prints its readings rather than setting them on a scale
- Bahnschrift's condensed widths are unused

The reviewer's "keep" line, which still holds: *keep the D10-to-D01 figure
exactly as honest as it is*. Refused leads stay full-height hatched ghost
plates with no reading. The line breaks at refusals rather than bridging
them. Every number comes from `/api/convergence`, and the whole ten-lead run
is shown.

## 5. Known limits (honest list)

- **Bahnschrift is a Windows face.** On macOS and Linux the sign stack falls
  back to "DIN Alternate", "Roboto Condensed", "Arial Narrow" or system-ui.
  The look degrades gracefully, but it is not identical.
- **The landing is chart paper only.** That is deliberate (the contract's
  ground). The working views have both themes.
- **The 3-D staff is desktop only.** Phones get the axis in words instead.
- **The volume view does not apply `lead` or `region` from the URL.** It
  carries them onward but opens on all lead days with no readout. Applying
  them would change its first view and its fly-through behaviour, which
  needs its own design.
- **F15** (volume frame rate on software rendering) is untouched.
- **Captures are local.** Like the baseline, they are not committed
  (2.9 MB per set).
