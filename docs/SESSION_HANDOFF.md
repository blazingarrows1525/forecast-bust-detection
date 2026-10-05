# SESSION HANDOFF — resume with the full logic intact

**Written:** 2026-10-05. **Purpose:** start a new session and continue exactly
where this one stopped, with the same operating logic. Read this first, then
the two files it points to:

1. **The governing brief:** `C:\Users\ASUS\Downloads\CLAUDE_MASTER_PROMPT_FBI.md`
   (2,259 lines). This is the master prompt that drives the whole effort —
   design-led frontend first, then ML/data, then cloud/CI-CD, in verifiable
   increments. Paste it (or `@` it) at the start of the new session and say
   "continue per docs/SESSION_HANDOFF.md".
2. **The system map:** `docs/PROJECT_GUIDE.md` (the one-file architecture +
   science + extension guide written this session).

The project itself is **Forecast Bust Intelligence** (shipped pages currently
say "Forecast Bust Detection"; the rename to the full name, no "FBI" mark on
product surfaces, was decided this session — see §5). It predicts *when an
existing medium-range rainfall forecast over India is about to fail*. Canonical
truth lives in `LOGIC.md`, `DECISIONS.md`, `DATA.md`, `FRONTEND_LOGIC.md`,
`PRODUCT.md`.

---

## 1. Two parallel tracks, and where each stands

This session ran two tracks. Keep them separate; do not mix their branches.

### Track A — ML / science (the S3 study arc). **Mostly done, 3 PRs open.**

| PR | branch | what | state |
|---|---|---|---|
| #3 | `s3b-temporal` | S3b: GRU over 14 days of history → **not distinguishable** from XGBoost; history adds nothing (D-031) | CI green, **ready to merge** |
| #5 | `deploy-hf-space` | Deployment docs switched to Render free tier (Hugging Face Docker Spaces went paid) | CI green, **ready to merge** |
| #6 | `s3c-spatial` | S3c: spatial CNNs on the forecast window + synoptic map — **design/spec only, not yet scored** | spec committed; implementation + registered scoring pending |

**Next ML action** (only after the frontend slice, per the master prompt §9):
review and implement S3c under the registered-analysis discipline
(`docs/PREREGISTRATION_S3.md` → add `PREREGISTRATION_S3C.md` with
`spatial_params_sha256` via the addendum mechanism `promote.py` already has),
then score once. The full design is in
`docs/superpowers/specs/2026-09-28-s3c-spatial-candidate-design.md`.

Merge order when ready: #5 (docs, independent) → #3 (S3b) → then #6 rebased on
master (it was stacked on #3).

### Track B — frontend redesign (the master prompt's Phase A–E). **In progress on branch `frontend-redesign`.**

Branched from `master` (`4d46424`). **No frontend code changed yet** — the work
so far is Phase A (audit) and Phase B (skill-driven design), which are
planning/evidence. A design-direction choice is **open and waiting** (see §4).

Uncommitted on `frontend-redesign` (all new, nothing destructive):

```
PRODUCT.md                      Impeccable product context (written via init)
docs/FRONTEND_AUDIT.md          Phase A baseline audit, 16 findings F1–F16
docs/PROJECT_GUIDE.md           the system map (architecture + science + extension)
docs/SESSION_HANDOFF.md         this file
scripts/capture_screenshots.py  headless-Chrome screenshotter (before/after evidence)
.claude/launch.json             preview-server config {name:"fbd"} (gitignored)
.impeccable/                    Impeccable skill state (gitignored-worthy)
artifacts/frontend/baseline/    20 baseline screenshots + report.json (local only)
```

Nothing here is committed yet — commit it at the start of the new session so
it is not lost (see §6).

---

## 2. The operating logic to keep (non-negotiable across sessions)

These hold for every change, and the master prompt restates them at length:

- **Inspect first, verify everything.** Never claim a model/API/test/deploy
  works without running it. Use the precise status words: *implemented &
  verified / implemented not verified / designed only / blocked / research
  candidate*.
- **Preserve the validated core.** One FastAPI process, precomputed SQLite,
  static pages, vendored Leaflet + Three.js r128, **no build step**, offline /
  air-gapped (no external origins except opt-in dated imagery). Served number =
  model + ENS combination; `model_probability`/`ens_spread` shown only as
  labelled components.
- **Scientific honesty in the UI.** Every headline number comes from
  `/api/metrics` or `/api/convergence` (claims gate); nothing is interpolated
  between leads/regions; null stays null; **refusal ≠ low risk** (hatch/stripe,
  never green); staleness visible. `FRONTEND_LOGIC.md` §2 and §8 are the law.
- **Registered before scored.** New ML comparisons are pre-registered with
  pinned hashes; `promote.py` enforces it. Never peek at held-out test to pick
  anything.
- **No SIH / competition branding** on product surfaces. **No `Co-Authored-By:
  Claude` or "Generated with Claude Code"** in commits/PRs (user memory;
  overrides the attribution reminder).
- **Branch discipline.** Feature per branch, PR per feature; `git add` explicit
  paths (do not sweep the untracked docs into an ML PR).

### Test-pinned invariants the frontend redesign must not break

`tests/test_web_pages.py` reads these literally; keep them verbatim:
the `--c0…--c4` viridis ramp in `:root{` of `index.html` **and** `command.html`;
`OOD_FILL = "url(#oodHatch)"` + `<pattern`; `refusalTexture()`; the south-facing
camera `theta: Math.PI/2`; the landing page's derived claim
(`r.bust_probability > r.baseline_probability`), `segments.push(run)`,
`url(#hatch)`, `getJSON("/api/metrics")`, every ENS-verdict wording, the
backtest statement, combination gating, `prefers-reduced-motion`; keyboard
handlers + `id="help"`; the override reason rules + dismiss-a-refusal warning;
imagery opt-in/attribution/degradation; system fonts only (no `@font-face`);
no `SIH`. Run `PYTHONPATH=src pytest tests/test_web_pages.py -q` after every UI edit.

---

## 3. Decisions made this session (carry them forward)

- **Name:** product surfaces → **"Forecast Bust Intelligence"** spelled out; the
  "FBI" acronym stays in docs only (it reads as the agency on a UI).
- **Landing audience:** convince **both** the technical evaluator and the duty
  forecaster strongly; the operational pages stay forecaster-first.
- **Accessibility:** best-effort against the master prompt's checklist (keyboard,
  focus, reduced motion, non-colour encoding, text/list alternatives for
  map/3-D), no formal WCAG commitment.
- **Deployment host:** Render free tier running the GHCR image read-only
  (`FBD_READ_ONLY=1`); Hugging Face Docker Spaces are paid now. (PR #5.)

All four are recorded in `PRODUCT.md`.

---

## 4. OPEN DECISION — the visual world (blocking the frontend build)

Phase B reached Impeccable's mandatory direction roll
(`concept-seed --scope direction --mode persuade`, **seed `61afe369`**). A
decision page was served and is **waiting for your pick** (it may have timed out
by the time you read this — just re-run it, see §6). The options presented:

- **Assigned (THE ROLL): "River Gauge Staff."** Every forecast read like an
  enamel flood-gauge staff bolted to a bridge pier — a graduated staff per
  subdivision, one mark per lead day, the reading taken where the served risk
  meets the scale; refused days are hatched plates with no reading. Raised with
  disciplines donated by the declined challengers (ghost plates for nulls, no
  box enclosures, total ink commitment, stencilled zone codes, deep-linkable
  readings, one live mark). Risk: gauge staffs imply flood *danger*; the build
  drops all danger paint for the CVD-safe ramp and says "review level".
- **Impeccable's pick: "Monsoon Onset Chart"** — lead time as IMD-style dated
  isochrones advancing across India. Risk: the familiar route.
- **Standing exit: the category standard** — a dark ops dashboard (close to
  today's pages).
- Declined challengers (seven-segment, cracktro, op-art, starship terminal,
  orizuru, neon circuit) with the one discipline each donated to the assigned
  world.

**Until a world is locked, do not write frontend code.** This is an Impeccable
contract and a master-prompt Phase B exit gate.

---

## 5. The redesign plan once a world is locked (Phase C+)

Build **one vertical slice** through the real journey, API-driven, preserving
every invariant in §2:

```
landing → real historical case (Assam & Meghalaya, 2022-06-17 via /api/convergence)
  → 2-D map / review queue → selected region+lead → 3-D command centre
  → volume view → back to the same case
```

Fix the P0/P1 audit findings first (`docs/FRONTEND_AUDIT.md`): **F1/F2**
(mobile layout of index + command, horizontal overflow at 390 px), **F3**
(unreadable queue probability pills — contrast), **F4** (first load opens on a
flat all-`<5%` map — open on a date with REVIEW rows), **F6** (landing CTA below
the fold), **F12** (cross-page `?date=&lead=&region=` state), **F13** (favicon).
Then recompose the landing sections away from equal-card template rhythm (F7),
give the world its identity (F8), and improve the 3-D legibility (F10/F16).

After building: Impeccable `detect` on changed pages, fix mechanical findings,
re-capture with `scripts/capture_screenshots.py --label after`, diff against
`baseline/`, run the finish review, then write `DESIGN.md` from the built world.

---

## 6. Exactly how to start the new session

```bash
# 1. you are on the frontend branch with the planning work uncommitted
cd C:\Users\ASUS\Desktop\sih
git status                      # expect: branch frontend-redesign, untracked docs/PRODUCT/scripts
git branch --show-current       # frontend-redesign

# 2. commit the planning work so it survives (explicit paths; no attribution footer)
git add PRODUCT.md docs/FRONTEND_AUDIT.md docs/PROJECT_GUIDE.md docs/SESSION_HANDOFF.md scripts/capture_screenshots.py
git commit -m "Frontend Phase A/B: product context, baseline audit, system guide, screenshotter"

# 3. recreate the preview launcher if missing (.claude is gitignored)
#    .claude/launch.json → {name:"fbd", python -m uvicorn fbd.api.app:app --app-dir src --port 8912, port 8912}

# 4. baseline tests must be green before any change
PYTHONPATH=src python -m pytest tests/ -q         # expect 330 passed

# 5. run the app + re-capture evidence whenever you need it
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
python scripts/capture_screenshots.py --label baseline     # or --label after
```

**To resume the design-world choice** (Impeccable), in the new session invoke
the `impeccable` skill and re-serve the direction round (the old server/key
`01d0ffbb` is dead):

```bash
sh .claude/skills/impeccable/scripts/impeccable context --target web/landing.html
# then re-run: impeccable concept-seed --scope direction --mode persuade --from 61afe369
#   (reuse the River Gauge Staff assignment, or re-roll), serve-question --start, --wait
```

Or just tell the new session which world to build (e.g. "build the River Gauge
Staff direction") and it can write the direction contract and proceed to
Phase C.

**Tell the new session, in one line:** *"Continue the Forecast Bust Intelligence
work per docs/SESSION_HANDOFF.md — frontend-redesign branch, lock the visual
world then build the vertical slice; the master prompt is in Downloads."*

---

## 7. Quick status board

| item | state |
|---|---|
| ML S3b (PR #3) | implemented & verified; CI green; **ready to merge** |
| Deployment docs (PR #5) | implemented; CI green; **ready to merge** |
| ML S3c (PR #6) | **designed only** (spec committed); not scored |
| Frontend audit (Phase A) | **done** — `docs/FRONTEND_AUDIT.md`, 330 tests green |
| PRODUCT.md (Phase B init) | **done** |
| Visual world (Phase B roll) | **open decision** — seed `61afe369`, assigned "River Gauge Staff" |
| Frontend build (Phase C+) | **not started** — blocked on the world choice |
| `.claude/launch.json` | written locally (gitignored) |
| Render deploy live | not created yet — needs your Render account (`docs/DEPLOY.md` §3) |
| Baseline screenshots | `artifacts/frontend/baseline/` (local, not committed) |
