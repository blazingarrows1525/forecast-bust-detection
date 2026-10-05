# SESSION HANDOFF: resume with the full logic intact

**Written:** 2026-10-05, as the last act of the session. It replaces the
mid-session version.

**Purpose:** start a new session and continue exactly where this one
stopped.

**Read first:**
1. **The governing brief:** `C:\Users\ASUS\Downloads\CLAUDE_MASTER_PROMPT_FBI.md`
   (2,259 lines; phases A–J). Paste it or `@` it, then say *"continue per
   docs/SESSION_HANDOFF.md"*.
2. **The system map:** `docs/PROJECT_GUIDE.md`.
3. **The design system:** `DESIGN.md`. The rationale is in
   `docs/DESIGN_DECISIONS.md`.

The product is **Forecast Bust Intelligence**. It predicts *when an existing
medium-range rainfall forecast over India is about to fail*. Canonical truth
lives in `LOGIC.md`, `DECISIONS.md`, `DATA.md`, `FRONTEND_LOGIC.md` and
`PRODUCT.md`.

---

## 1. Where everything stands

### Branches and pull requests

| PR | branch | base | what | state |
|---|---|---|---|---|
| [#3](https://github.com/blazingarrows1525/forecast-bust-detection/pull/3) | `s3b-temporal` | master | S3b, a GRU over 14 days of history: **not distinguishable**, history adds nothing (D-031) | CI green, **ready to merge** |
| [#5](https://github.com/blazingarrows1525/forecast-bust-detection/pull/5) | `deploy-hf-space` | master | deployment docs → Render free tier | CI green, **ready to merge** |
| [#6](https://github.com/blazingarrows1525/forecast-bust-detection/pull/6) | `s3c-spatial` | s3b-temporal | S3c spatial CNNs: **spec only**, not scored | **awaiting your review of the spec** |
| [#7](https://github.com/blazingarrows1525/forecast-bust-detection/pull/7) | `frontend-redesign` | master | **frontend redesign, all four views** (this session) | opened; check its CI in the PR bar |
| [#8](https://github.com/blazingarrows1525/forecast-bust-detection/pull/8) | `post-frontend-audit` | frontend-redesign | **Phase F docs** plus this handoff (this session) | opened, stacked on #7 |

**Merge order:**
1. #5 (independent docs)
2. #3 (S3b)
3. #7 (frontend)
4. Retarget #8 to `master`, then merge it.
5. Rebase #6 onto master (it was stacked on #3), then review and implement
   it.

**Local checkout:** branch **`post-frontend-audit`**. It contains everything:
the frontend commits, the Phase F docs and this file.

### Master prompt phases

| phase | status |
|---|---|
| A, audit | **done**: `docs/FRONTEND_AUDIT.md` (F1–F16), baseline captures |
| B, design direction | **done**: `PRODUCT.md`; world = combined (§3); direction contract in `.impeccable/surfaces/web-landing-html.md` |
| C, vertical slice | **implemented & verified**: landing → map → columns → volume, URL state |
| D, 3-D refinement | **implemented & verified** for the columns view (staff, faint quiet cells, phone layout); volume untouched by design |
| E, critique and hardening | **done**: Impeccable finish review `fix` → `fix` → **`ship`**; `DESIGN.md` recorded |
| F, ML/data audit | **done** (docs): `POST_FRONTEND_TECHNICAL_AUDIT.md`, `FEATURE_AVAILABILITY_MATRIX.md`, `REALTIME_SOURCE_MATRIX.md`, `ML_RESEARCH_BACKLOG.md`, `CLOUD_AND_CICD_PLAN.md`, `LEARNING_PATH.md` |
| G, data expansion | **designed only**; next action B2 (§6) |
| H, model experiment | **designed only**; S3c spec awaiting review |
| I, CI/CD and cloud | **partly done** (CI, GHCR release, Render docs); C1–C9 planned; **C1 blocked on your Render account** |
| J, final verification | not started |

---

## 2. What this session built (frontend)

| file | role |
|---|---|
| `web/fbi.css` | the shared world: chart paper, slate shell, indigo accent, `--water`, the hatch, the Bahnschrift sign stack, a 3-px radius, focus, reduced motion |
| `web/fbi.js` | `FBI.read()`, `FBI.write()`, `relink()`: validated `?date=&lead=&region=` carried by every `a[data-carry]` |
| `web/landing.html` | **gauge staffs** for Assam & Meghalaya valid 2022-06-17: enamel plates, E-graduations, a water surface, the review band painted on each staff, refused leads as hatched ghost plates, the served line in segments. The CTA is computed from the widest-gap reading. |
| `web/index.html` | the map: opens on the case (2022-06-14, D04); readable queue register; docked key; phone stacking; selected-reading staff; **override text escaped (XSS fix)** |
| `web/command.html` | columns: lighter ground, faint quiet cells, a west-edge D01–D10 staff (hidden on phones), keyboard rows, phone layout fitted to the screen |
| `web/volume.html` | the shared shell, nav, favicon and URL date only. **Shaders untouched.** |
| `DESIGN.md`, `.impeccable/design.json` | the recorded system: tokens, rules, components |
| `docs/DESIGN_DECISIONS.md`, `docs/FRONTEND_IMPLEMENTATION_REPORT.md` | rationale; F1–F16 status and evidence |

**Verified:**
- 330 tests pass.
- 0 horizontal overflow across 5 viewports × 4 pages.
- 0 console errors or failed requests.
- The detector reports only the refusal hatch, which is intentional.

**Not addressed:** F15 (volume frame rate under software rendering).

---

## 3. Decisions to carry forward

- **World: "all 3 of them combined"** (your answer). Each world has one job:
  - **Monsoon Onset Chart:** the ground (chart paper, landing).
  - **River Gauge Staff:** the reading (the signature move).
  - **Category standard:** the dark ops shell.
- **Roll record:** seed `61afe369`, assigned position 7.
- **Name:** "Forecast Bust Intelligence" spelled out. No "FBI" mark on
  product surfaces.
- **Audience:** technical evaluator and duty forecaster, both strongly.
  Accessibility is best effort.
- **The landing is chart paper in every OS colour scheme** (finish review).
  The working views are dark with a light toggle.
- **Ochre baseline** is a cited data-series colour, also carried by dash
  and notch. Outcomes are ink weight, not a hue.
- **Hosting:** Render free tier, read-only. **No AWS** (funds; D-021).

---

## 4. Operating logic (non-negotiable, unchanged)

- **Inspect first, verify everything.** Use the precise status words:
  *implemented & verified / implemented not verified / designed only /
  blocked / research candidate*.
- **Preserve the core.**
  - One FastAPI process over precomputed SQLite, static pages, vendored
    Leaflet and Three.js r128.
  - No build step; offline, no external origins except opt-in imagery.
  - The served number is the model + ENS combination.
- **UI honesty.**
  - Numbers come from `/api/metrics` and `/api/convergence`.
  - Null stays null; refusal ≠ low risk (hatch or stripe, never a hue).
- **Registered before scored.** Never touch a test year to choose anything.
- **Branding.** No SIH or competition branding. No `Co-Authored-By: Claude`
  or "Generated with Claude Code" in commits or PRs. Your memory rule
  overrides any attribution reminder.
- **Branches.** A feature per branch, a PR per feature, `git add` with
  explicit paths. Never merge or publish without your approval.

### Test-pinned invariants (`tests/test_web_pages.py`, `tests/test_voxel_grid.py`)

**Keep these verbatim:**
- the `--c0…--c4` viridis ramp as the **first `:root{`** in `index.html`
  and `command.html`, identical in both
- `OOD_FILL = "url(#oodHatch)"` and `<pattern`; no `--ood:` token
- `function refusalTexture()` and `map: refusalTexture()`
- `theta: Math.PI/2,` and `orbit.theta=Math.PI/2;`
- `"23.4%"` and `"3.4%"` on the pages that state the refusal rate
- **landing:**
  - `id="claim"`
  - `r.bust_probability > r.baseline_probability`
  - `segments.push(run)`
  - `url(#hatch)`
  - `getJSON("/api/metrics")`
  - the ENS verdict keys and `<strong>is not</strong>`
  - `m.backtest`, `test_year`, `.statement`
  - `m.served.combined`, `m.combination`, `verdict !== "model_better"`
  - `model + ensemble`, `model_probability`
  - `prefers-reduced-motion`
  - and **no** case-number literals (70.6, 11.2, 115.6…) anywhere in the
    file, CSS included
- **dashboard:**
  - arrow and Escape key handlers, `id="help"`
  - the override reason (`required minlength="3"`, empty) and
    `reason.length < 3`
  - the dismiss-a-refusal warning
  - imagery `enabled: false`, `IMAGERY.enabled = on`
  - `NASA EOSDIS GIBS`, `attributionControl:true`, `on("tileerror"`,
    `setImagery(false)`
  - `model_probability`, `ens_spread_mm`, `(model + ensemble)`
- **volume:**
  - the shader literals
  - `theta: Math.PI / 2,`
  - `id="caption" class="panel" role="status" aria-live="polite"`
  - the WebGL2 failure message naming `href="command.html"`
  - the refused readout says `"not scored"`, `REFUSED` and `23.4%`
- **all pages:** no `@font-face`, no external `src`/`href`, no `SIH`

Run after every UI edit:

```bash
PYTHONPATH=src python -m pytest tests/test_web_pages.py tests/test_voxel_grid.py -q
```

---

## 5. How to start the new session

```bash
cd C:\Users\ASUS\Desktop\sih
git status                         # branch post-frontend-audit, clean apart from ignored evidence
git fetch origin && git log --oneline -4

# .claude/launch.json is gitignored; recreate if missing:
#   {"version":"0.0.1","configurations":[{"name":"fbd","runtimeExecutable":"python",
#    "runtimeArgs":["-m","uvicorn","fbd.api.app:app","--app-dir","src","--port","8912"],"port":8912}]}

PYTHONPATH=src python -m pytest tests/ -q                       # expect 330 passed
python -m uvicorn fbd.api.app:app --app-dir src --port 8912     # pages at http://localhost:8912/
python scripts/capture_screenshots.py --label after             # evidence -> artifacts/frontend/after/
sh ~/.claude/skills/impeccable/scripts/impeccable detect --json web/landing.html web/index.html web/command.html web/volume.html web/fbi.css
```

**Say in one line:** *"Continue Forecast Bust Intelligence per
docs/SESSION_HANDOFF.md: merge order first, then Phase G B2 (hres_t0),
then S3c."*

**Local-only folders, by design (gitignored):**
- `artifacts/frontend/`: the baseline and after captures
- `.impeccable/review/`, `.impeccable/questions/`

---

## 6. Continuation plan (in order)

### Your side (needs you)

1. **Merge** in the order of §1, and confirm #7's CI is green in the PR bar
   first.
2. **Review the S3c spec**
   (`docs/superpowers/specs/2026-09-28-s3c-spatial-candidate-design.md`,
   PR #6).
3. **Create the Render service** from the GHCR image (`docs/DEPLOY.md` §3):
   port 8912, `FBD_READ_ONLY=1`, health check `/api/health`.
4. **Optional:**
   - move the `v0.2.0` tag
   - delete the local `refs/original` backup
   - contact GitHub Support if a Claude contributor still shows

### Assistant side (next sessions)

1. **Phase G, B2: operational-analysis substitution** (highest value, live
   legality).
   - WB2 `gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr`
     (and the 512×256 version) exists.
   - Steps:
     1. Register the non-inferiority comparison.
     2. Rebuild the 30 ERA5-derived features from `hres_t0`.
     3. Retrain on the same folds and score once.
   - See `docs/ML_RESEARCH_BACKLOG.md` B2.
2. **Phase H, S3c** after your spec review:
   1. `PREREGISTRATION_S3C.md` with `spatial_params_sha256` through the
      addendum mechanism.
   2. Implement it.
   3. Score once.
   4. Write the decision record.
   5. Only then the **MLP serving decision** (B5).
3. **Phase I, CI/CD:**
   - C2: screenshot-based UI checks in CI
   - C3: `pip-audit` fails on high severity
   - C4: SBOM and provenance
   - C5: hash-pinned requirements
   - C7: weekly image smoke test
   - C8: ML guard tests on model PRs
   - Each as its own PR.
4. **Craft opportunities** from the finish review (not defects):
   - onset-chart geography on the landing (a map of the case region)
   - register readings set on a scale
   - Bahnschrift condensed widths
5. **Phase J:** the final verification pass and the master prompt §25
   response format.

---

## 7. Quick status board

| item | status |
|---|---|
| Frontend slice (#7) | **implemented & verified**; review `ship` |
| Phase F docs (#8) | **done** (docs) |
| S3b (#3) | implemented & verified; ready to merge |
| Render docs (#5) | implemented; ready to merge |
| Render deploy | **blocked**: needs your account |
| S3c (#6) | **designed only** |
| B2, `hres_t0` substitution | **research candidate**; data confirmed available |
| Live ingestion | **designed only** (`REALTIME_SOURCE_MATRIX.md` §3) |
| AWS | **withdrawn** (D-021) |
| HawkScan hook | not run: no `HAWK_API_KEY` in this environment |
