# Serve the model + ENS combination — design

**Date:** 2026-09-25
**Status:** approved in chat (combination is the served number; three design sections)
**Follows:** D-029 — model + ENS spread outranks ENS spread alone on average over
2019–2021, +0.0244 [+0.0178, +0.0308]. `FRONTEND_LOGIC.md` §8 allows the claim
only once the product shows the combination.

---

## 1. What changes for a forecaster

The number on the map, in the review queue, in the escalation tier and in the
assistant's answers becomes the combination of the model and the 50-member
ensemble spread. Beside it the dashboard shows the two ingredients — the model
alone and the ENS spread — and the reasons gain a line saying what the ensemble
did to the risk. A day the model refuses is still refused.

## 2. Facts verified

| fact | source |
|---|---|
| The product serves precomputed `bulletins.sqlite`: 79,900 rows, 2021–2022 (splits val + test), frozen model v0.1.0 | store, `meta` |
| Store is gitignored and shipped as Release asset `v0.1.0`, pinned by SHA-256 | `scripts/fetch_release_artifacts.py` |
| ENS is complete on disk for 2021 and 2022 (122 dates each, leads 1–10) | `E.dates_by_year` |
| Consumers of `bust_probability`: API (`_to_prediction`, review queue, verification, risk cube, convergence), escalation, GenAI tools and guardrails, dashboard, landing, `replay_demo.py`, `plot_evaluation_figures.py` | grep |
| API data quality is computed per request (`_quality`), never read from the stored column | `app.py` |
| Reasons are plain English strings, top-3 TreeSHAP | `explain/reasons.py`, store |
| The landing convergence chart plots `bust_probability` against `baseline_probability`; its captions are computed from the data | `landing.html` |

## 3. The served combiner

- Fitted once by `scripts/fit_combiner.py` on **2021**, the frozen model's
  calibration year: logistic regression on [logit of the frozen model's
  **uncalibrated** probability, log(1 + ENS spread)] over 2021's Day 3–7 rows —
  D-029's recipe on the served model.
- Written to `data/artifacts/combiner.json` (committed): the three coefficients,
  the median log(1 + spread) of the fit rows (the reference for the reason line),
  fit year, row count, and the SHA-256 of the model it was fitted for.
- Applied to **all ten leads**. D-029's evidence covers Days 3–7; D-030 says so.
- Served by `fbd.model.combined` (pure numpy): the same arithmetic as the
  scikit-learn combiner, so serving needs no fitting library. A test proves the
  two agree.

## 4. The store, version 0.2.0

- `bust_probability` = the combination where ENS spread exists; otherwise the
  model's calibrated probability, and the row's stored `data_quality` is
  `ENS_UNAVAILABLE`. (None in the 2021–2022 archive; it matters for live data.)
- New columns, appended: `model_probability` (the model alone, calibrated) and
  `ens_spread` (mm/day).
- OOD refusal unchanged: a refused row carries no probability of any kind.
- Interval: each bagged model's raw score goes through the same combiner; the
  interval is re-centred on the served number, as before.
- Reasons: an ensemble line **first**, then the top-3 TreeSHAP lines. The line
  states the spread, whether it is high, low or typical against 2021's median,
  and its exact effect on the odds, from the combiner's log-odds contribution:
  typical if |contribution| < 0.1, else "raises/lowers the bust odds ×k".
- `meta`: `model_version` 0.2.0, `combined` = 1, the combiner coefficients.

## 5. API and pages

- `BustPrediction` and `ReviewQueueItem` gain `model_probability` and
  `ens_spread_mm` (both optional). `model_version` is read from the row.
- **Tolerant of the v0.1.0 store:** absent columns read as null, so a fresh clone
  serving the old Release asset still works; it just shows no combination.
- GenAI tools select the two new columns when present; their numbers become
  grounded automatically.
- `/api/metrics` gains `served` (`model_version`, `combined`) from the store's
  `meta`, and `combination` (`data/artifacts/combination.json`).
- **Landing:** the combination sentence ("the model and the ensemble together
  outrank the ensemble alone …", interval and years from `/api/metrics`) renders
  **only when `served.combined` is true** — §8's condition in code. The
  convergence chart's model line is the served number, labelled
  "model + ensemble" when combined.
- **Dashboard:** the detail panel says "probability this forecast busts (model +
  ensemble)" and adds "model alone X% · ensemble spread Y mm/day".

## 6. Knock-on

- Regenerate the store, then `plot_evaluation_figures.py` (reliability and
  refusal figures). Refusal counts must be unchanged; the served-subset ECE and
  top-bin figures change and are restated from the output.
- README: the demo case and the figure paragraph are rewritten from the new
  store; "What it does" says the served number combines both.
- D-030 records the served combination's 2022 Day 3–7 AUROC, Brier and ECE
  (descriptive: 2022 has been seen), the demo-case change, and the Day 1–2/8–10
  extrapolation.
- **Release asset:** the new store needs Release `v0.2.0` and a new pinned hash.
  Publishing is outward-facing: stop and ask. Until approved, the fetch script
  keeps pointing at `v0.1.0`.

## 7. Tests

| file | proves |
|---|---|
| `tests/test_combined.py` (CI) | pure apply = logistic arithmetic; ensemble reason wording for high / low / typical; missing spread falls back to the model; agreement with `fbd.evaluate.combine.Combiner` (skips without sklearn) |
| `tests/conftest.py` | fixture store in v0.2.0 shape; a v0.1.0 fixture for tolerance |
| `tests/test_model_api.py` | new fields present on v0.2.0, null on v0.1.0 |
| `tests/test_metrics_endpoint.py` | `served` block from `meta`; `combination` block |
| `tests/test_web_pages.py` | landing gates the combination sentence on `served.combined` and hardcodes none of its numbers; dashboard shows the two ingredients |

## 8. Commit order

```
1  fbd.model.combined + fit_combiner.py + combiner.json + tests
2  generate_bulletins v0.2.0 + fixtures
3  API, tools, metrics + tests
4  landing + dashboard + tests
5  regenerate store; figures; verify refusals unchanged
6  D-030, README, FRONTEND_LOGIC, HANDOFF; final audit; push
7  ask to publish Release v0.2.0; on yes, publish and pin
```
