# Learning path: learn the system by working in it

A sequence of modules tied to files that exist in this repository (master
prompt §18). Each module covers eight things:
1. **Why** it matters here
2. **Files** that implement it
3. **Concept** in plain English
4. **Inspect**: commands to inspect or test it
5. **Exercise**: one small, safe change
6. **Expect**: the expected output
7. **Pitfalls**: common failure modes
8. **Interview**: how to explain it

Commands assume the repository root, `PYTHONPATH=src`, and the server
running:

```bash
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
```

Read `docs/PROJECT_GUIDE.md` first for the map. Every number you quote in
an interview should come from `/api/metrics`, a test, or a decision record:
**no invented metrics**.

---

## Track 1: Frontend and data visualisation

### 1.1 The page contract: semantics, layout, URL state

- **Why.** Four static pages must work offline, on a phone, and hand a
  reading from one view to the next.
- **Files.** `web/fbi.css` (tokens, the top bar, controls), `web/fbi.js`
  (`FBI.read`, `FBI.write`, `relink`), `web/index.html`.
- **Concept.**
  - The URL is the state that travels.
  - Validate it like any input, write it with `replaceState` (no history
    spam), and rewrite links instead of storing state in the browser.
- **Inspect.**
  - Open `/?date=2022-06-14&lead=4&region=ASSAM_MEGHALAYA` and click
    "Columns".
  - Run `pytest tests/test_web_pages.py -q`.
- **Exercise.** Add `&view=baseline` support to `fbi.js` and `index.html`.
  Validate it against `{model, baseline}`.
- **Expect.** Reloading keeps the Baseline view; an invalid value is
  ignored.
- **Pitfalls.**
  - Trusting the query string (it ends up in `fetch` URLs and the DOM).
  - Using `pushState` in a scrub loop.
- **Interview.** "Deep-linkable, validated URL state replaced
  per-page state, so a forecaster can hand a colleague an exact reading."

### 1.2 Leaflet and GeoJSON with an honest refusal

- **Why.** The map is the product, and a refused region must never look
  low-risk.
- **Files.** `web/index.html`: `render()`, `colour()`, `installHatch()`,
  `OOD_FILL`.
- **Concept.**
  - Each polygon gets a fill from the viridis ramp.
  - A refused region gets an SVG *pattern* instead of a colour, because a
    hue would sit on the risk axis.
- **Inspect.**
  - Find a date with refusals: `/api/bulletin?init_date=2022-06-14`, filter
    `status != "OK"`.
  - Run `pytest tests/test_web_pages.py -k refusal`.
- **Exercise.** Change the hatch angle to 135° in `installHatch()`.
- **Expect.** The tests still pass, because the angle is not pinned but the
  pattern is.
- **Pitfalls.**
  - Leaflet draws into one SVG per pane: define the pattern once, and
    rebuild it on a theme change.
- **Interview.** "Refusal is encoded as texture, not colour; a colour-blind
  test computes CIELAB distances under simulated CVD."

### 1.3 Three.js: scene graph, picking, a sprite axis

- **Why.** The column view shows all ten lead days at once.
- **Files.** `web/command.html`: `buildMap()` (merged geometry),
  `buildColumns()`, `pick()` (Raycaster), `buildStaff()` (CanvasTexture
  sprite), `refusalTexture()`.
- **Concept.**
  - Merge static geometry into one mesh: about 1,330 objects become 2 draw
    calls.
  - Pick by casting a ray against the column meshes only.
  - A canvas texture gives you text in 3-D without a font loader.
- **Inspect.** In the browser console, run `colGroup.children.length`. The
  answer is 340 (34 × 10).
- **Exercise.** Make `quiet` cells (<5%) use `depthWrite:true` and compare
  how much of the map you can see.
- **Expect.** The map hides behind tall quiet columns. That is why
  `depthWrite` is off for them.
- **Pitfalls.**
  - Rebuilding all meshes on every change without disposing them (memory).
  - The camera orientation: `theta: Math.PI/2` is south; the test pins it.
- **Interview.** "Merged geometry, raycast picking, a CanvasTexture axis,
  and a keyboard path to the same selection."

### 1.4 Raymarched volume (read-only study)

- **Why.** It is the most technically dense view, and its shaders are
  test-pinned.
- **Files.** `web/volume.html` (shaders `PRELUDE`, `FRAG`, `PICK_FRAG`),
  `tests/test_voxel_grid.py`.
- **Concept.**
  - DDA voxel traversal with `texelFetch` (nearest-sampled, never
    interpolated).
  - The review flag is decided in JavaScript from the real probability,
    then packed into a bit.
- **Inspect.** Run `pytest tests/test_voxel_grid.py -q` and read why each
  assert exists.
- **Exercise.** None in the shaders. Change only the HUD copy.
- **Expect.** 31 tests still pass.
- **Pitfalls.** Using `LinearFilter` (it invents values between cells);
  bloom (it smears values).
- **Interview.** "The render never implies more resolution than the model
  has; tests enforce it."

### 1.5 Scientific colour and uncertainty display

- **Files.** `index.html` `:root{` ramp; `tests/test_web_pages.py`
  (`_simulate`, `MIN_ADJACENT_DE`); `landing.html` `drawGauge()`.
- **Concept.**
  - The colour ramp is chosen by measurement.
  - Uncertainty is shown as intervals with the number, and the review level
    is drawn on the scale.
- **Exercise.** Swap `--c3` for a red, run the test, and read the failure.
- **Expect.** A CVD separability failure naming the band pair.

---

## Track 2: Data science and ML

### 2.1 Temporal folds and leakage

- **Why.** Weather is autocorrelated; a random split lies.
- **Files.** `src/fbd/evaluate/folds.py`, `tests/test_fold_leakage.py`,
  `docs/FEATURE_AVAILABILITY_MATRIX.md`.
- **Concept.**
  - Each fold trains on earlier years, calibrates on the next year, and
    tests on the one after.
  - Every fitted statistic (P95, climatology, z-scores) uses training years
    only.
- **Inspect.** `pytest tests/test_fold_leakage.py -v`.
- **Exercise.** In a scratch copy, make `error_thresholds()` use all years.
  Run the test, read the failure, revert.
- **Expect.** `test_bust_thresholds_ignore_later_years` fails.
- **Pitfalls.** Standardising over every date. The project measured its own
  instance of this (the regime leak, +0.0009).
- **Interview.** "Expanding-window folds; leakage is tested, not trusted;
  the one disclosed leak was measured and reported."

### 2.2 The bust label

- **Files.** `src/fbd/labels/bust.py`, `tests/test_core.py`.
- **Concept.** A bust is a conjunction: a large error AND a category flip
  AND significant rain. That makes it rare (about 4%).
- **Exercise.** Compute the base rate per lead from `dataset.parquet` with
  pandas.
- **Expect.** The rate rises with lead.

### 2.3 XGBoost, isotonic calibration, Brier score

- **Files.** `src/fbd/model/train.py`, `scripts/train_model.py`,
  `src/fbd/evaluate/metrics.py`.
- **Concept.**
  - Gradient-boosted trees rank the risk.
  - Isotonic regression on a separate year turns the ranking into honest
    probabilities.
  - Brier skill and ECE judge the probabilities; AUROC judges the ranking.
- **Inspect.** `curl -s localhost:8912/api/metrics | python -m json.tool | head -60`.
- **Interview.** "Ranking and calibration are separate jobs, fitted on
  separate years."

### 2.4 OOD detection and the refusal policy

- **Files.** `src/fbd/ood/detector.py`, `src/fbd/quality/escalation.py`.
- **Concept.**
  - The Mahalanobis distance to the training cloud; above the training
    99.5th percentile, refuse.
  - A missing probability is REFUSE, never AUTO_OK.
- **Exercise.** Change `quantile` to 0.99 in a scratch run and count the
  refusals.
- **Expect.** More refusals. That is the tradeoff between coverage and
  honesty.
- **Interview.** "Refused days bust 23.4% vs 3.4%. Unknown is not safe, so
  the UI never draws it as safe."

### 2.5 Decision cost

- **Files.** `escalation.py:cost_optimal_threshold`.
- **Concept.** With a 10:1 miss-to-false-alarm cost, flag when
  p ≥ 1/(1+10) = 0.0909.

### 2.6 Registered experiments and intervals

- **Files.**
  - `docs/PREREGISTRATION_S3.md`
  - `src/fbd/evaluate/registration.py`, `promotion.py`
  - `scripts/promote.py`
  - `tests/test_backtest_guard.py`, `test_promotion.py`
- **Concept.**
  - Freeze the hypothesis and the hashes in a committed file; the scorer
    refuses otherwise.
  - Use a Bonferroni level across the slate.
  - Use a cluster bootstrap over init dates, not rows.
- **Exercise.** Edit one byte of a registration file and run `promote.py`.
- **Expect.** A `RegistrationError`.
- **Interview.** "I made p-hacking mechanically impossible in my own
  pipeline."

---

## Track 3: Data engineering

### 3.1 Sources, run identity, idempotent fetch

- **Files.**
  - `src/fbd/ingest/ens_fetch.py` (shards, `done_dates`, `ShortEnsemble`)
  - `scripts/fetch_ens.py`
  - `tests/test_fetch_ens_resume.py`
- **Concept.** One shard per init date, resumable. A short ensemble is
  refused, never averaged.
- **Inspect.** `pytest tests/test_fetch_ens_resume.py -v`.
- **Next.** `docs/REALTIME_SOURCE_MATRIX.md` §3 designs the live version.

### 3.2 Regridding and area weights

- **Files.** `src/fbd/regions/`, `weights_*.parquet`,
  `tests/test_core.py::test_area_mean_*`.
- **Concept.** Overlap is computed once in an equal-area projection
  (EPSG:7755). The area mean renormalises over valid cells and rejects
  thin coverage.

### 3.3 Storage and release artifacts

- **Files.**
  - `scripts/generate_bulletins.py` (SQLite store)
  - `scripts/fetch_release_artifacts.py` (checksum-pinned download)
  - `DATA.md` (what is in git and what is not)
- **Exercise.**
  `sqlite3 data/artifacts/bulletins.sqlite "select key,value from meta"`.

---

## Track 4: DevOps and MLOps

### 4.1 The container and the read-only mode

- **Files.** `Dockerfile` (multi-stage, non-root `fbd` user),
  `tests/test_read_only.py`, `docs/DEPLOY.md`.
- **Inspect.**
  ```bash
  docker build -t fbd .
  docker run -e FBD_READ_ONLY=1 -p 8912:8912 fbd
  ```
  then POST an override.
- **Expect.** A 403.

### 4.2 CI as a set of claims

- **Files.** `.github/workflows/ci.yml` (test, invariants, airgap,
  security, container), `publish.yml` (tag → smoke test → GHCR).
- **Concept.** Each job proves one claim. The airgap job imports the app
  with sockets blocked.
- **Next.** `docs/CLOUD_AND_CICD_PLAN.md` C2–C8.

### 4.3 Drift and monitoring

- **Files.** `src/fbd/mlops/drift.py` (PSI per feature),
  `src/fbd/quality/drift.py` (KS for the serving health check),
  `scripts/monitor_drift.py`, `/api/health` (`drift_status`), `/metrics`
  (Prometheus).
- **Concept.** PSI for the offline report, and KS where no binning
  agreement is possible. The archive build correctly reports DRIFT and
  STALE against today's clock.

---

## Portfolio narrative (from evidence only)

What you built, with the file or record that proves each item:

- **A decision-support system** that predicts forecast failure: the store,
  the API, four views.
- **A registered-analysis discipline:**
  - S1, S1b and S1c (D-025, D-026, D-029)
  - S3a and S3a-C (D-027, D-028)
  - S3b (D-031, PR #3)
- **The model + ensemble combination as the served number** (D-030).
- **An offline, air-gapped frontend** with a tested colour-vision-safe ramp
  and refusal encoding, deep-linkable state, and a reviewed design system
  (`DESIGN.md`).
- **Free-tier CI/CD to GHCR** with a smoke-tested release (`v0.2.0`).
