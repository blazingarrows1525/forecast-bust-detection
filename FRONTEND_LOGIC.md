# FRONTEND_LOGIC.md — what the interface must be true to

Companion to `LOGIC.md` (the modelling logic) and `DECISIONS.md` (the decision
log). This file is the **domain and invariant spec for anyone building UI**,
including an AI given a prompt. Read it before designing a screen.

If you only read one section, read **§2 Safety invariants**. Everything else is
craft; §2 is correctness.

---

## 1. What this system is, in one paragraph

It does **not** forecast the weather. IMD already does that. This is a
**meta-model over an existing forecast**: given a rainfall forecast that has
already been issued, it predicts whether *that forecast is about to fail*.
Output is per **IMD meteorological subdivision** (36 of them) per **lead day**
(1–10), for a given **init date** (the day the forecast was issued).

The user is a **duty forecaster at IMD** with roughly **45 minutes** before a
bulletin deadline. They are the authority. The system advises; it never issues,
suppresses, or escalates a warning. The UI must never imply otherwise.

**The value proposition in one sentence:** as a flood approached in June 2022,
the ensemble *converged* — which reads as growing confidence — while this model
went the other way and said the forecast was about to bust. It was right.

---

## 2. Safety invariants — non-negotiable

These are not style preferences. Violating one makes the interface dangerous.

### 2.1 A refusal is not a low-risk result

`status: OUT_OF_DISTRIBUTION` means **the model declined to score this
region-day** because the atmospheric state is unlike its training data.
`bust_probability` is `null`.

On the **2022 test year**, measured:

| | rows | actually busted |
|---|---|---|
| scored (`status: OK`) | 39,565 | **3.4%** |
| refused (`OUT_OF_DISTRIBUTION`) | 385 | **23.4%** |

**Refused days bust 6.9× more often than accepted ones.** A refusal is
*unknown, elevated* risk.

> **Rule:** a refused region must never render green, must never render in the
> same visual family as "low risk", and must never be visually quieter than a
> scored low-risk region. Use a distinct encoding — diagonal hatching, a
> cross-hatch, an outline treatment — that reads as *"not assessed"*, not as
> *"fine"*. Grey is the default mistake here: grey reads as "nothing to see".

This has already caused one real failure in this project: an LLM narrating a
row said *"the system declined to score it"* about a row whose status was `OK`
and whose bust probability was 1.000 — the highest-risk cell on the map. A
forecaster reading that stands down on the worst cell. See `DECISIONS.md`
D-019 addendum 3–4.

### 2.2 Three tiers, never two

`review_tier` is `AUTO_OK` / `REVIEW` / `REFUSE`. A binary "flagged / not
flagged" destroys the distinction in §2.1. `tier_guidance` carries plain-words
text for each — render it, do not paraphrase it.

The `REVIEW` threshold is **0.0909**, and it is not a free parameter: it is
`C_fa / (C_fa + C_miss)` with a 10:1 cost ratio (a missed bust costs 10× a
false alarm). If a judge asks why the threshold "seems low", that is the
answer, and it is arithmetic rather than taste.

### 2.3 Never invent or interpolate a number

Every number shown must come from the API. Do not smooth a probability across
lead days for a nicer animation, do not interpolate a missing lead, do not
average subdivisions for a prettier choropleth. If a value is `null`, show that
it is `null`.

### 2.4 No directive language

The UI advises review. It never says "evacuate", "issue a red alert", "close
schools", or what the public should do. Copy is addressed to a forecaster about
which forecasts merit a second look. This is enforced in code on the assistant
path (`fbd.genai.guardrails.check_authority`) and must hold in static copy too.

### 2.5 Staleness must be visible

`data_quality` is `OK` / `DEGRADED` / `STALE` / `UNAVAILABLE`, and
`input_age_hours` says how old the inputs are. The shipped build serves a
**2016–2022 reanalysis archive**, so `mode=live` reports `STALE` *by design* —
that is honest, not a bug. `banner` carries human-readable warning text; render
it across the top when present.

### 2.6 Uncertainty is part of the number

`prediction_interval` is `[lo, hi]`. `confidence_in_estimate` says how much to
trust the point estimate. A bare probability with no interval overstates what
the system knows. Note the model is **overconfident in its extreme tail** —
the highest bin predicts 0.207 and observes 0.158 (`docs/FIGURES.md`).

---

## 3. Hard technical constraints

### 3.1 Zero external origins — enforced by CI

`.github/workflows/ci.yml` has a job that **fails the build** if
`web/index.html` references any `http(s)://` origin in a `src` or `href`.

That kills: Tailwind CDN, Google Fonts, React/Vue from unpkg or jsdelivr,
Chart.js from a CDN, icon fonts, analytics, web fonts of any kind.

This is **deliberate and load-bearing**. The product's headline property is that
it runs with the network unplugged, in an air-gapped ops room. Do not trade it
for convenience.

**What to do instead:**
- Vendor into `web/vendor/` (already there: `leaflet.js`, `leaflet.css`,
  `three.min.js`).
- Use a **system font stack** — `system-ui, -apple-system, "Segoe UI", Roboto,
  sans-serif`.
- Inline SVG for icons and charts. No icon library.
- Plain CSS with custom properties. No preprocessor, no build step.
- If a framework is genuinely needed, vendor the production bundle and commit
  it — but weigh that against: the current app is **285 lines of HTML** and
  needs no toolchain at all. That simplicity is defensible in a way that a
  `node_modules` tree is not.

### 3.2 Offline and small

The serving image deliberately has no geo stack (D-020). Geometry is
precomputed: `/api/regions` returns a static, pre-simplified 0.64 MB GeoJSON.
Do not add a runtime geometry dependency.

### 3.3 Where the frontend lives

```
web/
  index.html      285 lines — the 2-D operational dashboard (Leaflet)
  command.html    529 lines — the 3-D risk cube (three.js r128)
  vendor/         leaflet.js, leaflet.css, three.min.js
```

Served by FastAPI at `/` (static mount). Run:

```bash
PYTHONPATH=src python -m uvicorn fbd.api.app:app --port 8912
```

---

## 4. API contract

Base: `http://localhost:8912`. Every endpoint is **read-only** except
`POST /api/override`. All dates are `YYYY-MM-DD` strings.

### `GET /api/health`
```jsonc
{
  "status": "ok",                       // ok | degraded | unavailable
  "model_version": "0.1.0",
  "model_loaded": true,
  "n_bulletin_rows": 79900,
  "init_date_range": ["2021-06-01", "2022-09-30"],
  "latest_init_date": "2022-09-30",
  "input_age_hours": 6.0,
  "data_quality": "OK",                 // OK | DEGRADED | STALE | UNAVAILABLE
  "drift_status": { "status": "OK" },   // OK | WATCH | DRIFT | UNKNOWN
  "notes": ["...render these..."]
}
```
Call this first. `latest_init_date` and `init_date_range` drive every picker.
**`status: "unavailable"` still returns HTTP 200** — check the field, not the
status code.

### `GET /api/replay/dates`
```jsonc
{ "init_dates": ["2021-06-01", "2021-06-02", ...] }   // 122 dates in 2022 alone
```

### `GET /api/regions`
GeoJSON `FeatureCollection`, 36 features. Properties:
`subdivision_id`, `name`, `n_districts`, `area_km2`.
**Join key is `subdivision_id` ↔ `region_id` elsewhere.**

### `GET /api/bulletin?init_date=&lead_day=&mode=replay|live`
Returns a `Bulletin`: `init_date`, `issued_at`, `lead_day`, `n_regions`,
`data_quality`, `input_age_hours`, `banner`, and `predictions[]` of
`BustPrediction`:

```jsonc
{
  "region": "Assam & Meghalaya", "region_id": "ASSAM_MEGHALAYA",
  "lead_day": 4, "init_date": "2022-06-14", "valid_date": "2022-06-18",
  "status": "OK",                       // OK | OUT_OF_DISTRIBUTION | CLIMATOLOGY_FALLBACK | UNAVAILABLE
  "bust_probability": 0.706,            // null unless status is OK
  "confidence_in_estimate": 0.62,
  "prediction_interval": [0.18, 0.74],  // may be null
  "dominant_factors": ["the forecast is +61.9 mm/day above ...", "..."],
  "regime": { "active_monsoon": 0.4, "monsoon_depression": 0.3, "entropy": 0.7, ... },
  "review_tier": "REVIEW",              // AUTO_OK | REVIEW | REFUSE
  "tier_guidance": "Elevated bust risk. Inspect ensemble products ...",
  "data_quality": "OK", "input_age_hours": 6.0, "ood_distance": 2.31,
  "baseline_probability": 0.112,        // the ensemble-spread baseline, same row
  "observed_rain_mm": 115.6, "forecast_rain_mm": 83.7, "actual_bust": 1,
  "model_version": "0.1.0"
}
```
`baseline_probability` exists **so the UI can toggle model vs baseline on the
same map** and a judge can see the gap directly. Use it.

`observed_rain_mm` / `actual_bust` are present for historical replay and absent
for a live forecast — that is the "show truth" toggle.

### `GET /api/review-queue?init_date=&top=`
Array of items, already ranked:
```jsonc
{ "rank": 1, "region": "Assam & Meghalaya", "region_id": "ASSAM_MEGHALAYA",
  "lead_day": 3, "valid_date": "2022-06-16", "bust_probability": 0.742,
  "status": "OK", "forecast_rain_mm": 110.5, "dominant_factors": ["...", "..."] }
```

### `GET /api/risk-cube?init_date=&mode=`
The whole space × lead-day field in one payload — **this is the endpoint for
animation and for the 3-D view.** No per-lead round trips.

```jsonc
{
  "init_date": "2022-06-14",
  "lead_days": [1,2,...,10], "decision_band": [3,4,5,6,7],
  "n_regions": 34, "data_quality": "OK", "input_age_hours": 6.0, "banner": null,
  "regions": [{
    "region_id": "ARUNACHAL_PRADESH", "region": "Arunachal Pradesh",
    "lon": 95.03, "lat": 28.06,
    "p":          [0.005, ...],   // arrays are PER LEAD DAY, index 0 = Day 1
    "status":     ["OK", ...],
    "actual":     [0, null, ...],
    "fcst_mm":    [2.76, ...],
    "obs_mm":     [15.2, ...],
    "valid_date": ["2022-06-14", ...]
  }]
}
```
**Every array is indexed by lead day, `index 0 = Day 1`.** `null` entries are
real — do not fill them.

### `GET /api/verification?init_date=&lead_day=`
Hindcast truth: `n_regions`, `n_busts_observed`, and `records[]` with
`bust_probability`, `baseline_probability`, `forecast_rain_mm`,
`observed_rain_mm`, `actual_bust`. Returns `{"detail": "no verification data"}`
when the valid date is past the archive.

### `GET /api/metrics` · `GET /metrics`
Model skill numbers, and Prometheus text respectively.

### `POST /api/override`
```jsonc
{ "region_id": "...", "init_date": "...", "lead_day": 4,
  "action": "dismiss" | "escalate", "reason": "min 3 chars", "user": "name" }
```
**Audited and immutable.** Every override is stored with user and reason so a
decision can be reconstructed after an event. If you build this UI, the reason
field is **mandatory** — that is the point of it.

---

## 5. The narrative the interface should tell

Use the real case. It is in the archive and every number below is served by the
API above.

**Assam & Meghalaya, June 2022 floods. Observed 115.6 mm/day.**

| Forecast issued | Lead | IMD forecast | **Our model** | Ensemble spread |
|---|---|---|---|---|
| 11 June | Day 7 | 86.7 mm | 45.6% | 33.3% |
| 12 June | Day 6 | 67.4 mm | 42.3% | 21.2% |
| 13 June | Day 5 | 74.7 mm | 47.5% | 17.6% |
| **14 June** | **Day 4** | **83.7 mm** | **70.6%** | **11.2%** |

As the event approached, **ensemble spread fell** — the models were converging,
which reads as growing confidence — **while our model rose**. The forecast
busted.

**Those two lines moving in opposite directions are the product.** Any hero
animation, landing page, or demo should be built on that divergence. It is one
chart and it needs no narration.

The Day-4 reason panel said, verbatim:
- the forecast is +61.9 mm/day above this subdivision's seasonal normal (top 1% for this subdivision)
- this subdivision and lead time historically bust often (5.0% of days)
- column moisture over India is below normal (−0.8 sd)

---

## 6. Design direction

### Colour
- Risk ramp: sequential, single-hue, **colour-blind safe**. Met agencies have
  colour-vision-deficient staff; this is an accessibility requirement, not a
  nicety.
- **Never green for refused** (§2.1). Refused gets a *pattern*, not a hue.
- Do not reuse IMD's official warning colours (red/orange/yellow alert). This
  system does not issue warnings, and borrowing that palette implies it does.
- Dark mode matters — this lives in an ops room.

### Motion
Animation must carry information, not decorate.

**Worth building:**
- **Lead-time scrub** (Day 10 → Day 1) with the map re-colouring. One
  `/api/risk-cube` call holds every frame.
- **The divergence chart** drawing in on scroll (`stroke-dasharray` +
  `IntersectionObserver`, ~40 lines, no library).
- **Forecast → observed reveal** on the verification view.
- **Reason waterfall** — `dominant_factors` as bars animating in.

**Do not build:**
- Odometer/count-up animation on probabilities. Animating a calibrated number
  toward its value is precision theatre, and this project has been careful to
  avoid exactly that.
- Anything that interpolates between lead days (§2.3).
- Parallax or scroll-jacking on an operational screen. The landing page can be
  cinematic; the dashboard cannot.

### Information density
A forecaster has 45 minutes. Favour density over whitespace on the dashboard.
Keyboard-first: arrow keys for lead day, `/` to search a subdivision, `Esc` to
clear. The landing page has the opposite job and may breathe.

---

## 7. What already exists, and what is missing

| | state |
|---|---|
| 2-D dashboard (`web/index.html`) | works: Leaflet map, date slider, lead picker, model/baseline/truth toggles, selected-region card, regime card, review queue, legend |
| 3-D risk cube (`web/command.html`) | works: three.js r128, geometry merged for a 4.9× draw-call reduction (1,669 → 343) |
| **Landing page** | **does not exist** |
| **Dark mode** | **does not exist** |
| **Keyboard navigation** | **does not exist** |
| **Animated lead-time scrub** | **does not exist** (the endpoint for it does) |
| **Divergence chart** | **does not exist** — highest value per line of code |
| **Override UI** | endpoint exists; no interface |
| Figures | `docs/figures/reliability.{png,svg}`, `earned_refusal.{png,svg}` |

---

## 8. Claims the UI may and may not make

Copy has to match what the evidence supports (`DECISIONS.md` D-022, D-025, D-026).

**May claim:**
- AUROC **0.840 [0.821, 0.859]** on a held-out year never used in training.
- Beats a cheap ensemble-spread proxy by **+0.082 AUROC [+0.062, +0.104]**.
- **Outranks a real 50-member operational ensemble over the 2022 held-out
  season:** **+0.0316 AUROC [+0.0141, +0.0485]** on 120 init dates (122 ENS
  dates fetched), from a test registered before the data was fetched (D-025).
  Always with the interval, the date count, **the year**, and beside it the
  backtest: not distinguishable over 2019–2021, **+0.0036 [−0.0059, +0.0127]**
  (D-026).
- Beats the cheap lagged proxy in 2022 and on average over 2019–2021,
  **+0.0364 [+0.0278, +0.0452]** (D-026).
- Refused days bust **6.9×** more often than accepted ones (23.4% vs 3.4%).
- Calibrated: ECE **0.0107**, with a stated overconfidence in the extreme tail.

**May NOT claim:**
- That the model outranks a real ensemble in general, or in any year but 2022.
  Over 2019–2021 the margin is **+0.0036 [−0.0059, +0.0127]**, and in 2019
  ENS spread outranks the model, **−0.0196 [−0.0359, −0.0031]** (D-026).
- That the edge holds in every part of the season. Late-season months favour
  the ensemble in several years (exploratory, D-025, D-026).
- That the model replaces the ensemble. A model + spread combination beats the
  model alone (+0.0154 [+0.0073, +0.0240], D-025 secondary b).
- That the product runs a neural network, or any transformer, BERT, LSTM,
  Random Forest or LightGBM. **None is served.** The served model is XGBoost +
  isotonic calibration + TreeSHAP + a Mahalanobis OOD detector. An MLP outranks
  the served XGBoost in a registered backtest across 2019–2022 (**+0.0088
  [+0.0023, +0.0157]**, D-027), but it is not served, and XGBoost is the better
  of the two in 2022. Claiming otherwise fails the first question a judge asks.
- That the MLP outranks a real ensemble. Its lead over ENS spread in 2019–2021
  was an uncorrected secondary; the registered confirmation on 2018 did not
  confirm it, **−0.0080 [−0.0254, +0.0087]** (D-028).
- Real-time operation. The build serves a 2016–2022 archive.
