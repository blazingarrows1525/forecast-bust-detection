# HANDOFF — Forecast Bust Detection (SIH26079)

**⚠️ READ THIS BEFORE STARTING WORK ON THIS REPOSITORY ⚠️**
This is a mature, benchmark-beating ML project for the Indian monsoon. Before writing any code:
1. **Understand the current state:** the project spans GenAI, MLOps, CI and Terraform, but with strict boundaries. Read the Verification Ledger (Section 3).
2. **Respect the contracts:** `LOGIC.md` is the locked engineering contract — do not silently deviate from it. `DECISIONS.md` holds the decision log (D-001 through D-025). `PROJECT_BLUEPRINT.md` is the long-form strategy.
3. **Run the sanity check:** `PYTHONPATH=src pytest tests/ -v` (expect 297 passed) and `PYTHONPATH=src python scripts/monitor_drift.py` should both come back green.
4. **Pick the next task** from Section 5.

---

## 1. Where the Project Stands

The system is a fully functional, end-to-end operational meta-model that predicts **when an existing medium-range rainfall forecast over India is about to fail (bust)**. 
Since the initial build, the project has expanded significantly in scope, adding an MLOps drift monitor, Prometheus observability, CI pipelines, Terraform IaC, and a heavily guardrailed GenAI assistant.

### Headline Results on Held-Out Test Year (2022)
**Day 3–7 Decision Band, 20,060 subdivision-days, bust base rate 3.31%.**  
Every predictor is evaluated under the *same* Isotonic Calibration fitted on the 2021 validation year:

- **True 50-member IFS ENS comparison (D-025, settled):** over the full 2022 held-out season (20,060 decision-band rows, 120 init dates) the model outranks raw ENS spread by **+0.0316 AUROC [+0.0141, +0.0485]**, on a test registered before the data was fetched (`docs/PREREGISTRATION_S1.md`). One season only; the edge sits in June–July, and a model + spread combination beats the model alone, so the ensemble is not redundant. The earlier 6,732-row, 40-date figures (+0.025 raw, +0.040 calibrated) are superseded. **It does not replicate outside 2022** (D-026, rolling-origin backtest, registered): averaged over 2019–2021 the margin is +0.0036 [−0.0059, +0.0127], and in 2019 ENS spread outranks the model. The claim that holds every year is the one over the lagged proxy.
- **Overall Model vs Spread Proxy:** AUROC 0.8400, Brier 0.0291, BSS +0.088, ECE 0.0108, Cost 237.3/1000.
- **Model + ensemble together (D-029, registered):** a logistic combination of the model's uncalibrated probability and ENS spread, fitted per fold on its validation year, outranks ENS spread alone on average over 2019–2021, **+0.0244 [+0.0178, +0.0308]**, and beats both parts in every year. This is the first repeatable edge over a real ensemble. The served product does not compute it yet.
- **Other model families (D-027, registered):** an MLP on the same 52 features outranks the XGBoost model across 2019–2022, **+0.0088 [+0.0023, +0.0157]** at a Bonferroni 98.33% interval, but XGBoost is clearly better in 2022. The MLP is not served. Two more declared candidates, temporal (S3b) and spatial (S3c), are to be scored under the same rule (`docs/PREREGISTRATION_S3.md`). Its apparent lead over the ensemble was **not confirmed** on 2018, the one untouched season (D-028): MLP − ENS spread −0.0080 [−0.0254, +0.0087].
- **Tests:** **297 tests pass** covering causality poison tests, exact partitions, bust invariants, GenAI guardrails, and drift monitors.
- **Offline Serving:** FastAPI backend with an interactive Leaflet dashboard (vendored JS/CSS). Air-gap validated.

---

## 2. The Expanded Platform Architecture (D-015, D-016, D-017)

1. **GenAI Assistant (`src/fbd/genai/`)**:
   - A completely offline-default, read-only AI assistant for forecasters. Features numeric-grounding guardrails (cannot invent bust probabilities) and non-interference rules.
   - Built using Anthropic Bedrock and local BM25 RAG (pure stdlib). 
   - **Important:** Enabled *only* via `FBD_GENAI_ENABLED=1`. Routes (`/api/assistant/*`) are not even registered without this flag.

2. **MLOps Drift Monitoring (`src/fbd/mlops/drift.py`)**:
   - Computes PSI per feature, prediction distribution shifts, calibration ratios, and OOD refusal rates.
   - **Key Finding:** JJAS 2022 had genuinely atypical upper-level wind shear (`india_shear_z` PSI 2.64). The held-out year is a *harder* test than assumed, reinforcing the generalisation claim.
   - Fast KS-based health checks exposed on `/api/health`.

3. **Observability (`src/fbd/obs/metrics.py`)**:
   - Exposes `/metrics` in Prometheus text format (no dependencies) and JSON structured logging.
   - Tracks `fbd_guardrail_violations_total` as a safety signal.

4. **CI/CD & Infrastructure (`infra/terraform/`, `.github/workflows/ci.yml`)**:
   - GitHub Actions CI workflow implemented (tests, decision-log invariants, air-gap, supply chain).
   - Terraform IaC for VPC/ALB/Fargate/ECR/CloudWatch deployment with strictly scoped Bedrock policies. 

---

## 3. The Verification Ledger — Read Before Claiming Anything

Do not claim what has not been run. The verification boundaries are explicit (D-017):

| Component | Status |
|---|---|
| Full pipeline, end to end | **Executed**, reproduces D-001..D-014 |
| 297 tests | **Executed**, 297/297 (25 Sep 2026) |
| Drift monitor | **Executed**, findings in D-016 |
| Docker + container health + dashboard air-gap | **Executed**, verified in a browser |
| GenAI guardrails / retrieval / tools / agent | **Executed** against a fake client |
| CI gates | **Executed locally**; never run on GitHub Actions |
| Any real LLM call | **EXECUTED** 19 Sep 2026 -- local, zero cost (D-019). Default is now `llama3.1:8b`, chosen on a measured fabrication rate (D-019 addendum 3) |
| LLM free-form narration | **EXECUTED and FAILED** -- fabricated a false status on a high-risk cell. The status-grounding guardrail now blocks it 6/6 on both models (D-019 addendum 4); narration still OFF, because a phrase list is evidence about the cases it covers and silence about the rest |
| Any real *cloud* LLM call | **NEVER** -- no credentials, no funding; local provider used instead |
| Terraform | **WITHDRAWN** -- never applied or validated; AWS path dropped for cost (D-021) |
| ECR push / ECS deploy | **WITHDRAWN** -- replaced by GHCR + free-tier hosting (D-021, `docs/DEPLOY.md`) |
| Container build + smoke test in CI | **AUTOMATED**, not yet run on GitHub Actions |

---

## 4. The Full Pipeline in Execution Order

All raw datasets (HRES, ERA5, IMD, true ENS) are locally cached in `data/raw/`. 
The entire pipeline reproduces from scratch in ~3 minutes.

```bash
# 1. Spatial aggregation definition: 36 IMD subdivisions from 641 districts
PYTHONPATH=src python -m fbd.regions.build

# 2. Ingest IMD gridded observations (2016-2022) & build subdivision truth
PYTHONPATH=src python scripts/fetch_imd.py
PYTHONPATH=src python scripts/build_truth.py

# 3. Ingest HRES forecasts, ERA5 initial state, and True ENS
PYTHONPATH=src python scripts/fetch_hres.py
PYTHONPATH=src python scripts/fetch_era5.py
PYTHONPATH=src python scripts/fetch_ens.py --year 2022 --every 3

# 4. Feature engineering & dataset construction -> data/processed/dataset.parquet (279,650 rows x 93 cols)
PYTHONPATH=src python scripts/build_dataset.py

# 5. Train baselines, XGBoost meta-model & calibration -> data/artifacts/bust_model.joblib
PYTHONPATH=src python scripts/train_model.py

# 6. Generate SQLite bulletins with bagged uncertainty intervals & TreeSHAP reasons
PYTHONPATH=src python scripts/generate_bulletins.py

# 7. Verification, MLOps, ablation, and stress testing
PYTHONPATH=src python scripts/ablation.py
PYTHONPATH=src python scripts/stress_test.py
PYTHONPATH=src python scripts/evaluate_ens_baseline.py --decision-band-only
PYTHONPATH=src python scripts/monitor_drift.py
PYTHONPATH=src pytest tests/ -v

# 8. Terminal demo & API server (Offline mode)
PYTHONPATH=src uvicorn fbd.api.app:app --port 8912
# Or with GenAI enabled:
FBD_GENAI_ENABLED=1 FBD_GENAI_TOOLS=1 FBD_GENAI_RAG=1 PYTHONPATH=src uvicorn fbd.api.app:app --port 8912
```

---

## 5. Next Steps for a Winning SIH Entry (from `PROJECT_BLUEPRINT.md`)

If you are picking this up to continue, focus on the high-leverage tasks detailed in the blueprint's Section 9:

1. **Multi-model AI ensembling (~2h):** Add GraphCast / Pangu / GenCast disagreement features. This is the killer novelty angle.
2. **Retrospective Narrative Case Studies (~3h):** e.g., Uttarakhand cloudburst 15 Aug 2022, Chennai floods 15 Nov 2021.
3. **Map Uncertainty Visualizations (~4h):** Make OOD subdivisions visually distinct (diagonal hatch), show 90% prediction intervals, and add a regime-strip mini-chart.
4. **PDF Bulletin Export (~2h):** Generate real, printable IMD-style reports (`/api/bulletin/pdf`).
5. **Cost-Sensitivity Slider (~2h):** Allow the UI to slide between 3:1 and 30:1 miss/false-alarm costs dynamically.

---

## 6. First Commands in a New Session

```bash
# 1. Check full test suite health (Expect 297/297)
PYTHONPATH=src pytest tests/ -v

# 2. Check the drift monitor
PYTHONPATH=src python scripts/monitor_drift.py

# 3. Start local API and Dashboard (offline defaults)
docker compose up -d
curl http://localhost:8912/api/health
```
