# Immersive frontend and research assistant: report

**Date:** 2026-10-18. **Brief:** the two master prompts of this round (premium 3-D
frontend, sections 1–19; web-grounded research assistant, section 20).
**Branch:** `live-path-pby8cg`.

> **Status after merging with `master` (2026-10-18).** Only the research
> assistant half of this report merged (D-039). The frontend half overlapped
> PR #26 (D-038) on the same pages and did not merge. D-038's frontend is
> served, with the assistant's "Ask" link added. This round's frontend is
> preserved on branch `frontend-round-chart-paper`. The frontend sections below
> describe that branch, not what is served.

Labels are used exactly: **IMPLEMENTED** (in the code), **TESTED** (a test or a
recorded browser run proves it), **UNVERIFIED** (written, not proven here),
**PROPOSED** (not built).

Three decisions were taken with you before any code, because the briefs
conflicted with earlier decisions:

| question | your answer | effect |
|---|---|---|
| Dark globe landing, or the chart-paper world from the earlier round? | Chart paper + 3-D reveal | The landing stays light chart paper with the gauge staffs; the opening is a real-time scene drawn on that paper |
| How far to take the web assistant, given the offline default and D-019? | Opt-in, off by default | Routes exist only with `FBD_GENAI_ENABLED=1 FBD_RESEARCH=1`; web only with `FBD_RESEARCH_WEB=1` |
| Keys? | Available on request | Built and tested against mocks; real calls stay UNVERIFIED until keys are added as environment secrets |

The brief names a competition; per the standing rule, no product surface carries
competition branding (`tests/test_web_pages.py` enforces it).

---

## A. Baseline audit

**Pages and navigation.** Case (`landing.html`), Map (`index.html`), Columns
(`command.html`), Volume (`volume.html`), 404. A shared nav on each; the reading
travels in the URL (`?date&lead&region`, `fbi.js`).

**Data-backed vs illustrative.** Every figure was already data-backed:
`/api/convergence` (case staffs), `/api/metrics` (register, ENS claim),
`/api/bulletin` + `/api/review-queue` (map), `/api/risk-cube` (columns),
`/api/voxel-grid` (volume), `/api/regions` (outlines). The volume's outline
smoothing is labelled. Nothing illustrative was presented as data.

**Protected invariants** (test-pinned, all still pass): the viridis ramp as the
first `:root{`; refusal as hatch/stripe, never a hue; south-side camera; nearest
sampling in the volume; no external origins except opt-in GIBS imagery; the
override form's required, never-prefilled reason; the landing derives every
number from the API; reduced motion respected.

**Defects confirmed in the baseline captures** (`scripts/capture_screenshots.py --label baseline`):

| # | defect | where |
|---|---|---|
| 1 | Review-queue chip and value wrapped onto two lines at 1440 px | Map, Columns |
| 2 | Phone header took about 360 of 844 px before the map | Map |
| 3 | "refused" plate labels collided with "77%" on phones | Landing |
| 4 | No motion or 3-D opening; the hero was a static figure | Landing |
| 5 | 344 draw calls, one mesh per cell | Columns |
| 6 | No WebGL failure path (a blank stage) | Columns |
| 7 | Archive replay visible only inside the status text | Map, Columns |
| 8 | Lead "scrub" ran D10 to D1 only, with no pause, direction, speed or reset | Map |
| 9 | Map and Columns could not stay in step while both were open | all |
| 10 | No "why this number?" view of a row's provenance | Map, Columns |

**Needs data or backend work before it can be shown honestly** (see E):
measured wind fields for any particle or Coriolis layer (the archive has no
U/V); a live mode (C9); multi-frame satellite crossfades (GIBS is daily).

## B. Design direction

* **Art direction.** "A scientific instrument with cinematic entry", inside the
  three worlds already chosen: chart paper (the ground), the river-gauge staff
  (the reading), and the dark ops shell (the working views). No globe, no glass,
  no neon.
* **Type and colour.** Unchanged system fonts (Bahnschrift/DIN signage for
  readings, tabular numerals); one indigo accent; viridis for risk everywhere,
  including the new opening; the hatch for refusal.
* **Controls.** Solid machined plates: a lit top lip and a shadowed foot, a 1 px
  lift on hover (pointer devices only), pressed flat on activation; selected,
  disabled, busy (`aria-busy`), done and error (`data-state`) states taken from
  attributes that assistive technology also reads. The primary action alone gets
  the solid accent.
* **Motion.** Three durations, one curve. Motion explains a change (plot,
  settle, tilt, rise; a stage completing), never loops for decoration, and
  disappears under reduced motion without losing any information.

## C. Changes implemented

### Shared — `web/fbi.css`, `web/fbi.js` (IMPLEMENTED, TESTED)
* Tactile control states, motion and spacing tokens, segmented controls.
* `.mode` "Archive replay" badge with the archive's date range (defect 7).
* Viewport-clamped tooltips: 180 ms on hover, immediate on keyboard focus,
  never on touch, dismissed by Esc (`title` lifted into `data-tip`).
* **Lead timeline**: exact steps D10…D01, play/pause (space), direction, speed
  0.5/1/2×, reset, valid date; pauses in a hidden tab (defect 8).
* **Live sync** across open views over a same-origin `BroadcastChannel`; remote
  changes are applied without re-broadcast (defect 9). TESTED in a two-tab
  browser run: region and lead map→columns, region and date columns→map.
* **"Why this number?"**: 20 provenance fields from the row and `/api/metrics`;
  missing values say "not provided by this build" (defect 10).

### Map — `web/index.html` (IMPLEMENTED, TESTED)
Timeline replaces the lead select and scrub; queue values no longer wrap
(defect 1); phone header packs into three rows with a sideways tool row, so
India and its legend fit the first screen (defect 2); a failed start says what
to do; "Ask" entry.

### Columns — `web/command.html` (IMPLEMENTED, TESTED)
* Five `InstancedMesh` buckets allocated once and refilled in place (defect 5);
  instance colours on a white material; hatch texture for refusals.
* Hover outline (one prebuilt `LineSegments`, moved, never reallocated).
* "Focus review" (f), render-quality tiers (auto/high/balanced/low/2-D only),
  live draw-call readout, WebGL-failure handover to the map (defect 6).

### Landing — `web/landing.html` (IMPLEMENTED, TESTED)
* The question first: "A forecast is about to fail. Which one?"; CTAs "Enter the
  command centre" and "Explore a verified historical case".
* **Opening (about 4 s, skippable)**: vendored three.js loaded after first paint;
  India plotted north to south as a pen recorder would; subdivisions settle; the
  camera tilts; every subdivision's served probability rises from
  `/api/risk-cube` (viridis, height = probability); refusals hatched at full
  scale height; the review level ringed on columns that reach it. The label is
  the widest gap among the leads that **busted** (Assam & Meghalaya, D04, served
  52.3% vs baseline 11.2%, busted), and the caption states that rule. Any click,
  key, wheel, touch or scroll finishes it; reduced motion starts finished;
  without WebGL a flat locator is drawn (defect 4).
* The gauge staffs open the case section; on narrow screens "refused" is
  lettered up the plate (defect 3).

### Research assistant (section 20) — IMPLEMENTED, TESTED with mocks; real providers UNVERIFIED
`src/fbd/genai/research/` (settings, schemas, router, urlsafe, extract,
providers, evidence, prompt, validate, state, pipeline, routes), `web/assistant.html`,
`docs/RESEARCH_ASSISTANT.md`, `docs/assistant-provider-decision.md`,
`tests/test_research_assistant.py`, `tests/assistant_eval/cases.jsonl`,
`scripts/evaluate_assistant.py`. Highlights:
* Deterministic router (7 intents plus an "act" rule: "send an alert" gets
  "read-only"); asks for a place rather than guessing; never asks for location.
* Archive facts and forecast-minus-observed arithmetic are read and templated,
  never narrated by a model (D-019).
* SSRF-safe fetching: scheme, credential and port rules; every resolved address
  checked; the connection pinned to the checked IP (DNS rebinding); three
  re-validated redirects; byte, time and content-type caps.
* Evidence policy by allowlist; injection-flagged pages shown but withheld from
  the model (two independent layers, each mutation-tested).
* Validation: unknown citation ids removed; any number absent from the sources,
  or any directive, withholds the whole answer.
* SSE with numbered events and real stages; no fake token streaming; Stop,
  retry, copy, clear; quotas; a TTL search cache; 30-minute in-memory follow-up
  memory; logs without message text; same-origin chat.

**Data-backed vs illustrative.** Everything shown is data-backed; nothing
illustrative was added.

## D. Verification

| check | command | result |
|---|---|---|
| Full suite | `PYTHONPATH=src python -m pytest tests/ -q` | **482 passed, 4 skipped** (3 need torch, which this environment's proxy blocks; 1 needs the raw shapefile) |
| CI's list | the `ci.yml` test list | **439 passed, 2 skipped**, including the two new assistant files |
| Assistant eval (mock) | `python scripts/evaluate_assistant.py` | **19/19**: route, status, citation validity, schema validity all 1.0 |
| Bandit (CI threshold) | `bandit -r src/ -ll` | no issues |
| Captures | `python scripts/capture_screenshots.py --label final` | 20 captures, 5 viewports × 4 routes, **0 console errors, 0 failed requests, 0 horizontal overflow** (`check_captures.py`: OK) |
| Reduced motion | emulated `prefers-reduced-motion: reduce` | opening starts finished, no skip control, no headline stagger |
| No WebGL | `--disable-webgl` | landing draws the flat locator; Columns hands over to the map |
| Live sync | two tabs, map + columns | region, lead and date follow in both directions |
| Assistant page | driven in a browser against the real store | archive explanation, method question, comparison: correct, cited; disabled build shows the off state |

**Scene measurements** (`python scripts/measure_scenes.py`; SwiftShader CPU
rasteriser, so frame times are relative, not what a GPU does):

| scene | draw calls | scene objects | frame | GPU growth after repeated view churn |
|---|---|---|---|---|
| Columns, before | 344 | 353 | 1.65 ms | 0 |
| Columns, after | **7** | **18** | **0.40 ms** | 0 |
| Landing opening | 5 | 9 | 0.52 ms | 0 |

Not measured: real-GPU frame rates, battery, and low-end phones. **Not
performed:** any deployment; any real web search or hosted-LLM call (keys absent,
hosts unreachable from this container); HawkScan (`HAWK_API_KEY` not set).

## E. Remaining work

**Required before a demo**
* Add `GEMINI_API_KEY` and `TAVILY_API_KEY` as environment secrets, allow their
  hosts in the environment's network policy, run
  `python scripts/evaluate_assistant.py --live`, and record the result in
  `docs/assistant-provider-decision.md` (closes the UNVERIFIED rows). Confirm the
  Gemini model id against the official models page.
* Watch the opening once on the demo machine's GPU and projector.

**Recommended next**
* PROPOSED: a Content-Security-Policy and security headers in the app (the pages
  use inline scripts, so this needs nonces or hashes to be meaningful).
* PROPOSED: quality tiers on the Volume view; the timeline on Volume (it has its
  own slice control today).
* PROPOSED: a provenance panel on the Volume readout.

**Optional experiments**
* PROPOSED: an atmospheric layer. The archive has no wind field, so any particles
  would be illustrative only and must be labelled "ILLUSTRATIVE PHYSICS"
  (f = 2Ω sin φ). Recommended only after the demo, behind a toggle, and only if
  it explains something a forecaster needs.
* PROPOSED: a feature-contribution waterfall. Store v0.3.0's reasons are
  occlusion, which is not additive, so a waterfall would misstate them; the
  reasons stay a list until an additive method is served.

## F. Career learning notes

* **WebGL scene management** — `web/command.html` `makeBuckets`/`buildColumns`:
  allocate once, update buffers in place (`setMatrixAt`, `setColorAt`,
  `needsUpdate`), and why `frustumCulled=false` is needed for instanced
  buckets. A real r128 trap fixed on the landing: `setColorAt` sizes its buffer
  from the *current* `count`, so fill it at full capacity first. Measure with
  `renderer.info` (`scripts/measure_scenes.py`).
* **Render on demand** — both 3-D pages draw only when something changed; the
  browser stops `requestAnimationFrame` in hidden tabs for free.
* **State synchronisation** — one canonical reading in the URL (`FBI.write`),
  broadcast to other tabs; apply remote changes without echoing them back
  (`applying` flag), and make every handler a no-op when nothing changed.
* **Data visualisation honesty** — the landing labels the reading by a stated
  rule (and was changed when the "widest gap" turned out to be a forecast that
  held); null is never zero (`spread not recorded`); refusals are present at
  full height.
* **Accessibility** — reduced motion as a first-class path, tooltips on focus,
  `aria-busy`, a live region for stages, a screen-reader summary of the 3-D
  scene, and a 2-D path for every 3-D value.
* **Backend safety for LLM features** — deterministic routing; server-issued
  source ids; validation that withholds instead of repairing; SSRF defences
  (resolve, check, pin); prompt-injection as data; quotas and same-origin
  checks against cost abuse. Read `src/fbd/genai/research/pipeline.py` top to
  bottom, then `tests/test_research_assistant.py`.
* **Testing practice** — mutation-test your guards: remove one, confirm a test
  fails (done here for citations and both injection layers). Add an eval case
  for every failure found in use (`tests/assistant_eval/cases.jsonl`).
* **Commands**:
  ```bash
  PYTHONPATH=src python -m pytest tests/ -q
  python scripts/capture_screenshots.py --label mine --base http://localhost:8912
  python scripts/measure_scenes.py --label mine --base http://localhost:8912
  python scripts/evaluate_assistant.py
  FBD_GENAI_ENABLED=1 FBD_RESEARCH=1 FBD_ASSISTANT_LLM=none \
    PYTHONPATH=src python -m uvicorn fbd.api.app:app --app-dir src --port 8912
  ```
