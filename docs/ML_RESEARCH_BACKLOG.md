# ML research backlog

What to study next, in order, with the registration each needs.
Written 2026-10-05 from `docs/POST_FRONTEND_TECHNICAL_AUDIT.md`.

**Rule for every item:**
1. Register before scoring (`docs/PREREGISTRATION_*.md` plus pinned
   hashes; `scripts/promote.py` addenda).
2. Use the existing folds (`src/fbd/evaluate/folds.py`).
3. Score once.
4. Never touch a test year to choose anything.

**Status words** (master prompt §0.2): designed only, research candidate,
in progress, implemented & verified, blocked.

## B1. S3c, spatial CNNs: complete the registered slate

- **Status:** implemented & verified, scored once (2026-10-06, D-032).
  **Promoted:** spatial − XGBoost +0.0159 [+0.0092, +0.0228]; spatial − MLP
  +0.0072 [+0.0047, +0.0097]. The forecast window carries the gain (window-only
  +0.0180, level with XGBoost in 2022); the map adds nothing detectable.
  Trained on the GPU, deterministically. Not served.
- **Hypothesis:** the *spatial pattern* of the forecast window and the
  synoptic map carries bust information that subdivision means discard.
- **Data:** HRES precipitation windows and ERA5/`hres_t0` fields on the
  0.703° and 1.5° grids, already fetched.
- **Registration:** `PREREGISTRATION_S3C.md` with `spatial_params_sha256`,
  through the addendum mechanism `promote.py` already has (the S3b
  pattern).
- **Comparison:** the frozen XGBoost fold models, read against the MLP so
  any gain is attributed to information, not to architecture. Bonferroni
  0.05/3.
- **Why first:** it closes the S3 slate. **The MLP serving decision (B5)
  waits on it** (D-027).
- **Acceptance:** one registered score, a decision record (D-03x) and the
  figure.

## B2. Operational-analysis substitution: make the model live-legal

- **Status:** implemented & verified, scored once (2026-10-06, D-033).
  **Inconclusive.** The TCWV was derived from `hres_t0` humidity and passed
  both gates (G1 RMSE 0.86, G2 RMSE 1.57 kg m⁻²). The margin was −0.0033
  [−0.0057, −0.0010] against 0.005: a small, real cost, so non-inferiority
  is not established. A live model on the operational state must say so.
- **Hypothesis:** rebuilding the 30 ERA5-derived inputs from the
  operational initial conditions costs **no material skill**. This is a
  non-inferiority question.
- **Registration:**
  - a non-inferiority margin (for example −0.005 AUROC), pinned before
    scoring
  - the same folds and labels
  - the same isotonic year per fold
- **Why it matters:** 30 of the 52 served inputs are reanalysis stand-ins
  (`FEATURE_AVAILABILITY_MATRIX.md`). Without this, no live prediction can
  honestly use them.
- **Acceptance:** non-inferior means `hres_t0` features become the live
  feature set. Inferior means the live path ships a reduced model and says
  so.

## B3. Tail calibration

- **Status:** implemented & verified, scored once (2026-10-06, D-035).
  **Adopt** at the next store regeneration:
  - The beta-family combiner narrows the top-decile reliability gap by
    −0.0091 [−0.0124, −0.0037], with the Brier score not worse.
  - The gain is mostly in 2019–2020. 2022 is barely helped, and its Brier
    score is slightly worse.
  - Venn–Abers was not tested.

## B4. Day 1–2 combination fit

- **Status:** implemented & verified, scored once (2026-10-06, D-036).
  **Adopt** at the next store regeneration. A Day 1–2-fitted combiner
  outranks the served extrapolation, +0.0023 [+0.0008, +0.0039], and removes
  the 2022 short-lead cost.

## B5. Serve a network?

- **Status:** implemented & verified, scored once (2026-10-06, D-034).
  Spatial + ENS outranks the served XGBoost + ENS: **+0.0064 [+0.0027,
  +0.0100]**, with lower calibration error in every year. The served
  combination still wins 2022 (−0.0142), the case-study season.
- **Next:** the spatial model is **eligible, not served**. Serving it needs
  its own design:
  - reasons for a CNN
  - OOD inputs
  - offline store regeneration
  - the owner's decision on the 2022 trade-off
- **Context:** D-027 promoted the MLP under the S3 rule. D-028 did not
  confirm its ENS lead on untouched 2018. The served product is unchanged
  by design until S3c is scored.
- **Decision input:** S3c's result, then a registered combination test
  (MLP + ENS vs the current XGBoost + ENS) before any store regeneration.

## B6. Multi-model disagreement (GEFS)

- **Status:** designed only.
- **Hypothesis:** disagreement between IFS and GEFS flags busts that IFS
  ENS spread alone misses.
- **Data:** GEFS on AWS from 2017. It needs alignment to 0.703°, version
  breaks handled, and its own fetch adapter with fixture tests.
- **Registration:** a new study; 2018 is no longer untouched (used by
  S3a-C), so a truly untouched period may require new seasons (2023+).

## B7. Historical analogue retrieval (product feature)

- **Status:** designed only (master prompt §14.1).
- **Idea:** for a selected reading, show the five most similar past
  region-days in standardised feature space, with their outcomes.
- **Rule:** training years only, never test.
- **Risk:** analogues read as a forecast. They must be labelled "similar
  situations, not a prediction".

## B8. Cost-ratio sensitivity (presentation only)

- **Status:** designed only.
- **Idea:** REVIEW_THRESHOLD = 1/(1+10) comes from a 10:1 cost ratio. Let
  the forecaster see the queue at 5:1 and 20:1.
- **No registration needed:** it changes no model or claim, only the
  threshold the tier uses. The default stays 10:1.

## Not on the list, and why

- **Another architecture search on the same 52 inputs:** the MLP already
  answered "is it the architecture?"
- **Using IMERG as an input:** it is verification-time data
  (`REALTIME_SOURCE_MATRIX.md` §2).
- **Random splits:** forbidden (master prompt §10.1). Weather is
  temporally dependent.
