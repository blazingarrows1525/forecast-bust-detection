# FRONTEND_HANDOFF.md — start here

Companion to [`FRONTEND_LOGIC.md`](FRONTEND_LOGIC.md), which holds the domain
rules, the safety invariants and the full API contract. **This file is the
working brief:** how to get the thing running, what to build in what order,
what "done" means, and a ready-to-paste prompt for an AI assistant.

---

## 1. Get it running in three commands

```bash
python scripts/fetch_release_artifacts.py
```
```bash
pip install -r requirements-serve.txt
```
```bash
PYTHONPATH=src python -m uvicorn fbd.api.app:app --port 8912
```

Then open <http://localhost:8912>. The first command pulls the ~63 MB bulletin
store (checksum-verified); it is gitignored, so **a fresh clone has no data
until you run it**. If the dashboard loads but every panel is empty, that is
the cause — check `/api/health`, it will say `status: "unavailable"`.

Good init dates to develop against: **`2022-06-14`** (the demo case, dense with
signal), `2022-07-20`, `2022-08-15`. Avoid `2022-09-30` — it is the last date
in the archive, so most lead days come back `UNAVAILABLE` and the review queue
is empty.

---

## 2. The one constraint that will bite you

**CI fails the build if `web/index.html` references any external origin.**

```yaml
- name: Dashboard must reference no external origin
  run: |  # greps for src=/href= pointing at http(s):// and exits 1
```

No Tailwind CDN. No Google Fonts. No React from unpkg. No CDN chart library.
No icon fonts. This is deliberate — "runs with the network unplugged" is the
product's headline property, and an air-gapped IMD ops room is the target
deployment. Vendor into `web/vendor/` or inline it. System font stack only.

Before pushing any frontend change:

```bash
PYTHONPATH=src python -m pytest tests/test_command_centre.py -q
```

---

## 3. Build order

Ranked by value per unit of work. **#1 and #2 are the ones that matter.**

### 1. The divergence chart — highest value, ~40 lines

The single asset that explains the product without narration. Two lines moving
in opposite directions as a flood approaches:

| Issued | Lead | Our model | Ensemble spread |
|---|---|---|---|
| 11 Jun | Day 7 | 45.6% | 33.3% |
| 12 Jun | Day 6 | 42.3% | 21.2% |
| 13 Jun | Day 5 | 47.5% | 17.6% |
| **14 Jun** | **Day 4** | **70.6%** | **11.2%** |

Observed: **115.6 mm/day**. The forecast busted.

Inline SVG, `stroke-dasharray` + `stroke-dashoffset` to draw on scroll via
`IntersectionObserver`. No library. Data is live from
`/api/bulletin?init_date=…&lead_day=…` (`bust_probability` vs
`baseline_probability`) — do not hardcode it; hardcoded numbers drift from the
archive and a judge may ask you to change the date.

**Done when:** it draws on scroll, reads correctly at 360 px wide, and the
numbers come from the API.

### 2. Lead-time scrub on the map

Drag Day 10 → Day 1 and watch risk build. **One `/api/risk-cube?init_date=…`
call holds every frame** — all 10 leads × 34 regions arrive in one payload, so
this is pure client-side work, no request per frame.

Arrays are indexed by lead day, `index 0 = Day 1`. `null` entries are real —
render them as "no data", never interpolate (`FRONTEND_LOGIC.md` §2.3).

**Done when:** scrubbing is smooth at 60 fps, `OUT_OF_DISTRIBUTION` regions
stay visually distinct at every frame, and no value is invented between leads.

### 3. Landing page (`web/landing.html`)

Lead with the **problem**, not the product. Suggested spine:

1. **Hero** — "A forecast is about to fail. Which one?" Then the divergence
   chart from #1.
2. **What a bust is** — the three-condition definition, animated: magnitude ∧
   category flip ∧ operational significance. All three must hold.
3. **The refusal** — `docs/figures/earned_refusal.png` or a live rebuild of it.
   23.4% vs 3.4%. "A refusal is not a low-risk result."
4. **Calibration** — `docs/figures/reliability.png`. Include the overconfident
   top bin; do not crop it out.
5. **Try it** — link into the dashboard at `2022-06-14`.

**Done when:** it works with JS disabled (content visible, animation is
enhancement), passes the no-external-origin check, and every number on it is
one the project is allowed to claim (`FRONTEND_LOGIC.md` §8).

### 4. Dark mode

`prefers-color-scheme` plus a manual toggle persisted in `localStorage`. Define
colours as CSS custom properties on `:root` and redefine them under the dark
media query. The risk ramp must stay colour-blind safe in both themes and the
refusal pattern must stay distinguishable.

### 5. Keyboard navigation

Arrow keys = lead day. `/` = focus subdivision search. `Esc` = clear selection.
`?` = shortcut overlay. The user has 45 minutes; mousing through a dropdown for
every region is the wrong interaction.

### 6. Override UI

`POST /api/override` exists and has no interface. Dismiss / escalate with a
**mandatory** reason (min 3 chars) and user. Every override is audited and
immutable so a decision can be reconstructed after an event — the reason field
is the entire point, so do not make it optional or prefill it.

---

## 4. Definition of done, for any frontend change

- [ ] `pytest tests/test_command_centre.py -q` passes
- [ ] No `http(s)://` in `src`/`href` anywhere in `web/`
- [ ] Refused regions are visually distinct from low-risk ones, and are not
      green and not merely grey (`FRONTEND_LOGIC.md` §2.1)
- [ ] `null` probabilities render as "not assessed", never as 0
- [ ] `banner` and `notes` from `/api/health` are surfaced when present
- [ ] Usable at 360 px wide
- [ ] No copy that issues a directive (§2.4) or makes a claim from §8's
      "may NOT claim" list
- [ ] Works against `2022-06-14` **and** against `2022-09-30`, where most leads
      are `UNAVAILABLE` — the empty state is a real state

---

## 5. Prompt to paste into an AI assistant

Copy everything between the lines. Attach `FRONTEND_LOGIC.md` alongside it.

---

> You are building the frontend for **SIH26079 — AI-Based Forecast Bust
> Detection for Medium-Range Weather Forecasts** (Ministry of Earth Sciences).
>
> **What the system does.** It does not forecast weather. It is a meta-model
> over an existing IMD rainfall forecast that predicts whether *that forecast
> is about to fail*, per IMD subdivision (36) per lead day (1–10). The user is
> a duty forecaster with ~45 minutes before a bulletin deadline. They are the
> authority; the system advises and never issues or suppresses a warning.
>
> **Read `FRONTEND_LOGIC.md` first.** It contains the safety invariants, the
> full API contract with real response shapes, and the list of claims the UI is
> and is not allowed to make. Treat §2 as correctness, not style.
>
> **Four rules you must not break:**
> 1. A region with `status: "OUT_OF_DISTRIBUTION"` has `bust_probability:
>    null` and means *unknown, elevated* risk — refused days bust 23.4% of the
>    time against 3.4% for scored ones. Never render it green, never in the
>    same visual family as low risk, never quieter than a scored low-risk
>    region. Use a pattern (hatching), not a hue. Grey is the default mistake.
> 2. **Zero external origins.** CI fails the build if `web/index.html`
>    references any `http(s)://` in `src`/`href`. No Tailwind CDN, no Google
>    Fonts, no React from a CDN, no icon fonts. Vendor into `web/vendor/` or
>    inline. System font stack. The product's headline property is that it runs
>    with the network unplugged.
> 3. Never invent, interpolate or smooth a number. If the API returns `null`,
>    show that it is `null`.
> 4. No directive copy — never "evacuate", "issue an alert", or what the public
>    should do. Address a forecaster about which forecasts merit a second look.
>
> **The story to build around.** Assam & Meghalaya, June 2022 floods, observed
> 115.6 mm/day. As the event approached the ensemble spread *fell* (33.3% →
> 11.2%, models converging, which reads as confidence) while this model *rose*
> (45.6% → 70.6%). The forecast busted. Those two lines moving in opposite
> directions are the entire product.
>
> **Stack:** plain HTML/CSS/JS, no build step. Existing app is 285 lines and
> needs no toolchain — keep it that way unless there is a strong reason not to.
> Vendored already: Leaflet, three.js r128. Inline SVG for charts and icons.
>
> **Run it:** `python scripts/fetch_release_artifacts.py` then
> `PYTHONPATH=src python -m uvicorn fbd.api.app:app --port 8912`. Develop
> against `init_date=2022-06-14`. Test the empty state against `2022-09-30`.
>
> **Build, in this order:** [state which of §3's items you want]
>
> Design direction: colour-blind-safe sequential risk ramp; refusal as a
> pattern not a hue; do not reuse IMD's official red/orange/yellow warning
> colours (this system does not issue warnings); dark mode; information-dense
> dashboard but a landing page that may breathe. Animation must carry
> information — no count-up animations on probabilities, no interpolation
> between lead days, no scroll-jacking on operational screens.

---

## 6. Things that will trip you up

| Symptom | Cause |
|---|---|
| Every panel empty, HTTP 200 | `bulletins.sqlite` missing — run `scripts/fetch_release_artifacts.py`. `/api/health` returns 200 with `status: "unavailable"`. |
| `/api/regions` 503 | Precomputed GeoJSON missing — `python -m fbd.regions.build && python scripts/precompute_geo_assets.py` |
| Region won't join to the map | Join `feature.properties.subdivision_id` ↔ `region_id`. The names differ (`ASSAM_MEGHALAYA` vs `Assam & Meghalaya`). |
| Risk cube has 34 regions, map has 36 | Two subdivisions have no scored rows on some dates. Render them as "no data", do not drop them from the map. |
| `mode=live` says STALE | Correct. The build serves a 2016–2022 archive; `mode=replay` is the historical demo. |
| Port 8912 serves a stale page | A Docker container may be bound to it — `docker stop forecast-bust-detection`. This has happened before. |
| CI fails on an unrelated frontend change | The no-external-origin grep. Check for a font or CDN link you added. |

---

## 7. Where everything is

| | |
|---|---|
| Domain rules, API contract, invariants | [`FRONTEND_LOGIC.md`](FRONTEND_LOGIC.md) |
| Modelling logic | `LOGIC.md` |
| Decision log (why anything is the way it is) | `DECISIONS.md` |
| Evaluation figures + the honest reading | `docs/FIGURES.md` |
| Backend handoff | `HANDOFF.md` |
| Data policy (what is and is not in the repo) | `DATA.md` |
| API source | `src/fbd/api/app.py`, `src/fbd/api/schema.py` |
| Current dashboard | `web/index.html` |
| Current 3-D view | `web/command.html` |
