# SESSION HANDOVER — Forecast Bust Detection (SIH26079)
## The one file to open at the start of a new session

**Author:** Harsh Trivedi · `trivediharsh1505@gmail.com` · SRMIST Kattankulathur
**Repository:** `C:\Users\ASUS\Desktop\sih`
**Handover written:** 23 August 2026, late afternoon IST
**Last verified commit:** `d66d188` — 62/62 tests, container healthy, offline default intact

This file is the single briefing that lets a new Claude Code / Antigravity window continue this project without re-reading transcripts. It contains:

1. **Copy-paste prompts** for opening a new window (§0).
2. What the project is and what has been built (§1–§5).
3. Every decision made so far, with numbers (§6, and pointers to `DECISIONS.md`).
4. What is done, what is half-done, what is untouched (§7).
5. The exact steps that need you — the human — with what to click and what to expect (§8).
6. Traps not to fall back into (§9).

Read alongside (all in repo root):
- `LOGIC.md` — LOCKED engineering contract
- `DECISIONS.md` — D-001…D-017 with evidence
- `PROJECT_BLUEPRINT.md` — long-form project overview
- `HANDOFF.md` — short resume brief
- `README.md` — public writeup with the corrected headline table
- `AWS_GUARDRAILS_IMPLEMENTATION.md` + `AWS_GUARDRAILS_REFERENCE.md` — the hardening walkthrough (partially implemented, see §7)

---

## 0. Copy-paste prompts

### 0.1 Kickoff prompt for a fresh session

Paste this into a new Claude Code window opened in `C:\Users\ASUS\Desktop\sih`:

> Open `SESSION_HANDOVER.md` in this repo and read it end-to-end before doing anything else. Then read `LOGIC.md`, the last three entries of `DECISIONS.md` (D-015, D-016, D-017), and `PROJECT_BLUEPRINT.md` §9A. After that, run `git log --oneline | head -10`, `PYTHONPATH=src python -m pytest tests/ -q`, and `docker ps --filter name=forecast-bust-detection` to check the actual state matches the handover.
>
> When you are done reading, tell me: (a) how many tests pass, (b) whether the container is running, (c) which unchecked item in §7's "What's left" table you would tackle first and why, and (d) any drift between the handover file and what you actually see. Do NOT start work until I confirm.

### 0.2 Resume prompt (if the session was interrupted)

> Same repo, same handover. Read `SESSION_HANDOVER.md` §7 (the checklist) and `git log --oneline | head -8` to see how far we got. Then continue from the next unchecked item. Follow the LOGIC.md contract; if anything conflicts, stop and flag it rather than proceeding.

### 0.3 If you want to continue the hardening walkthrough specifically

> Continue `AWS_GUARDRAILS_IMPLEMENTATION.md` from Step 7 (explanation-stability check). Steps 1–6 are done and committed at `d66d188`. Read `SESSION_HANDOVER.md` §7 first for the exact status of each step, then work through the rest in order. Every 👤 step: stop, tell me exactly what to do and what result to expect, wait for me to confirm.

### 0.4 If you want to focus on SIH presentation instead of more code

> Read `SESSION_HANDOVER.md` and `PROJECT_BLUEPRINT.md` §9 (SIH strategy). Help me build the 90-second demo script and the judge-defence answers. Prioritise the "novelty framing that survives a hostile judge" and the "corrected headline claim" — those are the two lines I need to be able to say cold.

---

## 1. What this project is, in five sentences

Medium-range monsoon rainfall forecasts fail catastrophically maybe a dozen days a season. When they fail, AI and physics models fail *on the same days* — so ensembling more models does not save you. This project is a **meta-model** that watches a forecast being issued and predicts, per subdivision per lead day, whether it is about to bust — with a plain-language meteorological reason for every flag. On the held-out year 2022 it beats a real 50-member IFS ENS by **+0.025 AUROC** on identical rows (0.832 vs 0.807), and **+0.040 AUROC / −16.8% decision cost / 2× economic value** when both are calibrated. It serves offline from a 63 MB SQLite file — dashboard verified with `externalScripts: []`, zero non-localhost requests, wifi unplugged.

**We do not compete with IMD's forecast. We tell IMD's duty forecasters which forecasts to double-check.**

---

## 2. The headline numbers — memorise these

**Held-out 2022, Day 3–7 decision band, 20,060 subdivision-days, base rate 3.31%:**

| Predictor | AUROC | Brier | BSS | ECE | Cost/1k | Value |
|---|---|---|---|---|---|---|
| Climatology | 0.519 | 0.0321 | −0.005 | 0.013 | 330.5 | 0.000 |
| Forecast rainfall alone | 0.750 | 0.0311 | 0.028 | 0.014 | 273.0 | 0.174 |
| Logistic (spread+lead) | 0.754 | 0.0310 | 0.031 | 0.010 | 293.4 | 0.112 |
| Ensemble spread (lagged proxy) | 0.758 | 0.0310 | 0.031 | 0.011 | 293.5 | 0.112 |
| **Our model (XGBoost + isotonic)** | **0.840** | **0.0291** | **0.088** | **0.011** | **237.1** | **0.282** |
| **True IFS ENS (calibrated, subsample)** | **0.792** | **0.0310** | **+0.045** | 0.008 | **287.0** | **0.145** |

**Two claims to say out loud, in this order:**
1. *"On identical rows against the real 50-member IFS ensemble, we win by +0.025 AUROC raw (0.807 vs 0.832) and by +0.040 AUROC / −16.8% decision cost / 2× economic value when both are calibrated on training-year data."*
2. *"The +0.082 AUROC that appears in intermediate rows is against the lagged-ensemble *proxy* used across the full archive; the true-ensemble comparison is the honest number."*

Both are recorded as D-014 in `DECISIONS.md`. Never quote +0.082 without immediately adding the +0.025 caveat.

---

## 3. The architecture in one page

```
        IMD 0.25° gauge rainfall ─┐
        WB2 IFS HRES forecasts    ├─► area-weighted aggregation to
        WB2 ERA5 analysis         ┤    36 IMD subdivisions
        WB2 IFS ENS (subsample)   ─┘
                                    │
                                    ▼
                  BUST LABEL (percentile + category flip + significant rain)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        forecast features    ERA5 state features    regime classifier
                                    │
                                    ▼
                        XGBoost + isotonic calibration
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
              OOD detector     3-tier ladder     TreeSHAP
              refuse to score  AUTO_OK/REVIEW/    -> plain-English reasons
                               REFUSE
                                    │
                                    ▼
                    SQLite bulletins → FastAPI → Leaflet dashboard
                                    │
                                    ▼
                          DUTY FORECASTER (the authority)

    Optional and default-OFF:
      Bedrock (Claude Opus 5) --> guarded assistant (grounded, refuses invented
                                  numbers, no write/alerting tool)
      Drift monitor          --> /api/health, distinct from the mlops offline check
      Prometheus /metrics    --> counter/gauge/histogram in-process
      Terraform IaC          --> VPC/ALB/Fargate/ECR (WRITTEN, NEVER APPLIED)
```

**Air-gap guarantee (LOGIC.md §13):** the default serving path makes zero external network calls. Verified in a browser: Leaflet 1.9.4 vendored, 36 subdivision polygons rendered, 4 requests all to localhost. CI enforces this via a job that imports the app with `socket.connect` monkeypatched to raise.

---

## 4. Full repository map

```
LOGIC.md                    the LOCKED engineering contract
DECISIONS.md                D-001..D-017, every deviation with evidence
README.md                   public writeup, corrected headline table
HANDOFF.md                  the short resume-work brief
PROJECT_BLUEPRINT.md        long-form project overview + SIH strategy §9
SESSION_HANDOVER.md         THIS FILE
AWS_GUARDRAILS_IMPLEMENTATION.md  hardening walkthrough (partially done)
AWS_GUARDRAILS_REFERENCE.md       reference version of the same

Dockerfile                  offline image, python:3.11-slim
docker-compose.yml          single service, no volumes, no network deps
requirements.txt            full pipeline dependencies
requirements-serve.txt      serving-only (smaller, in the container)
requirements-genai.txt      optional Bedrock/boto3, NOT in the container

config/imd_subdivisions.json  641 districts -> 36 subdivisions
.github/workflows/ci.yml      5-job CI: tests, invariants, air-gap, security, container
infra/terraform/              VPC/ALB/Fargate/ECR — WRITTEN, NEVER APPLIED (D-017)

src/fbd/
  config.py                   central constants, bounding boxes, thresholds
  regions/build.py            36-subdivision constructor, exact-partition validator
  regions/masks.py            EPSG:7755 polygon-cell overlap weights
  ingest/wb2.py               WeatherBench 2 Zarr access
  ingest/imd.py               IMD 0.25° NetCDF -> subdivision area-means
  ingest/hres.py              cached HRES -> subdivision, paired with truth
  labels/bust.py              three-clause bust definition (LOGIC.md §4.3)
  features/forecast.py        lagged-ensemble spread proxy, jumpiness, anomalies
  features/era5.py            Somali jet, BoB vort., TCWV, shear (join on init_date)
  regime/classify.py          6 soft regime probabilities + entropy
  model/train.py              XGB depth 4 + isotonic, class-weighted
  model/baselines.py          climatology / rain-only / logistic / spread
  ood/detector.py             Mahalanobis, 99.5th percentile (earned refusal)
  explain/reasons.py          native TreeSHAP + family-deduped reasons
  evaluate/metrics.py         AUROC, Brier, BSS, ECE, cost, value
  api/schema.py               Pydantic outputs + ReviewTier enum
  api/app.py                  FastAPI service + observability middleware
  api/genai_routes.py         /api/assistant/* (mounted ONLY when flag on)

  quality/                    NEW in this session (steps 3-5 of the walkthrough):
    validate.py               physical-range gate reusing config.PHYSICAL_RANGES
    escalation.py             AUTO_OK / REVIEW / REFUSE, threshold from cost ratio
    drift.py                  KS-based drift check for the health endpoint

  mlops/drift.py              PSI/calibration/OOD retrain-verdict pass (D-016)
  obs/metrics.py              Prometheus registry + JSON structured logs (D-017)

  genai/                      NEW in this session (D-015, default OFF):
    settings.py               feature flags, all default false
    guardrails.py             numeric-grounding + non-interference + injection
    retrieval.py              BM25 over LOGIC/DECISIONS/README/BLUEPRINT/HANDOFF
    tools.py                  4 read-only tools over the bulletin store
    client.py                 AnthropicBedrockMantle with honest degradation
    agent.py                  hand-written tool loop so guardrails sit in the middle

web/index.html + web/vendor/leaflet.{js,css}   fully vendored dashboard

tests/
  test_core.py                12 tests: causality, partition, bust label
  test_model_api.py           9 tests: metrics, OOD, reasons, schemas
  test_genai.py               28 tests: guardrails, retrieval, agent loop, API surface
  test_mlops.py               13 tests: PSI, drift assessment, escalation semantics

scripts/
  fetch_imd.py, fetch_hres.py, fetch_era5.py, fetch_ens.py
  build_truth.py, build_dataset.py, train_model.py, generate_bulletins.py
  ablation.py, stress_test.py
  evaluate_ens_baseline.py    the D-014 baseline comparison
  monitor_drift.py            the mlops.drift retrain check
  replay_demo.py              terminal replay of the Assam & Meghalaya case

data/                         (all git-ignored except data/artifacts/*.csv and .joblib)
  raw/                        IMD NetCDF, HRES/ERA5/ENS Zarr caches
  processed/dataset.parquet   279,650 x 93, the modelling frame
  artifacts/
    bust_model.joblib         1.5 MB
    bulletins.sqlite          63 MB, 79,900 rows
    reference_distributions.npz  drift snapshot (NEW)
    results.json, ablation.csv, ens_*.csv, stress_*.csv, shap_importance.csv
    drift_report.json         from scripts/monitor_drift.py
```

---

## 5. How to run everything, in the order it makes sense

```bash
# 0. Environment
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

# 1. Full pipeline (~3 minutes off cached raw data)
PYTHONPATH=src python -m fbd.regions.build
PYTHONPATH=src python scripts/fetch_imd.py           # cached; skips if present
PYTHONPATH=src python scripts/build_truth.py
PYTHONPATH=src python scripts/fetch_hres.py          # ~9 GB transfer if uncached
PYTHONPATH=src python scripts/fetch_era5.py          # ~17 GB transfer if uncached
PYTHONPATH=src python scripts/build_dataset.py       # 20 s
PYTHONPATH=src python scripts/train_model.py         # 14 s
PYTHONPATH=src python scripts/generate_bulletins.py  # 130 s

# 2. Evidence
PYTHONPATH=src python scripts/ablation.py
PYTHONPATH=src python scripts/stress_test.py
PYTHONPATH=src python scripts/monitor_drift.py --decision-band-only
PYTHONPATH=src python -m pytest tests/ -v

# 3. True-ENS baseline (needs the parquets in data/raw/wb2/ens/)
PYTHONPATH=src python scripts/fetch_ens.py --year 2022 --every 3
PYTHONPATH=src python scripts/evaluate_ens_baseline.py --decision-band-only

# 4. Serve
docker compose up -d     # -> http://localhost:8912  (offline by default)
# stop with:
docker compose down
```

**Correct API URLs** (memorise the second one — `/api/bulletin/latest` is NOT a route, returns 422):

- `GET /api/health` — model version, row count, drift status, staleness banner
- `GET /api/bulletin?init_date=YYYY-MM-DD&lead_day=N&mode=replay|live` — all regions
- `GET /api/bulletin/{region_id}?init_date=YYYY-MM-DD` — one region across leads
- `GET /api/review-queue?init_date=YYYY-MM-DD&top=12` — ranked highest risk
- `GET /api/verification` — bust-rate tables
- `GET /api/metrics` — model benchmark JSON
- `GET /metrics` — Prometheus text-format telemetry (different thing)
- `POST /api/override` — logged with user/reason/timestamp
- Optional (flag ON only): `/api/assistant/{status,ask,search}`

---

## 6. The commits so far, in one line each

```
d66d188  Partial hardening pass: input validation, escalation ladder, serving-side drift
ab1e68a  Update PROJECT_BLUEPRINT with the expanded platform and verification ledger
ae2ca00  Add CI, Terraform IaC, optional GenAI requirements; record D-017
47fb55e  Add drift monitoring (MLOps); record D-016
e802a4f  Wire GenAI routes behind the flag; add observability
fe53156  Add guarded GenAI layer: Bedrock, RAG, tool calling, guardrails
94802b6  Refresh D-012 ablation table; record D-015 scope expansion
d3568ea  Baseline: verified-green state before full-scope expansion
```

Every commit is a rollback point. To undo an experiment safely:

```bash
git log --oneline                    # find the SHA before the mistake
git reset --hard <sha>               # DESTRUCTIVE — check `git status` first
# safer alternative:
git switch --detach <sha>            # look at the old state without losing anything
```

---

## 7. What is done, half-done, and untouched

### From `AWS_GUARDRAILS_IMPLEMENTATION.md`

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | 🤖 `git init` | ✅ Done | Baseline commit `d3568ea`. |
| 2 | 🤖 Open-Meteo divergence feature | ⏳ Not started | Highest-value remaining automation. Fetch, feature, ablation row. |
| 3 | 🤖 Input validation gate | 🟡 Half done | Module written (`src/fbd/quality/validate.py`), reuses `config.PHYSICAL_RANGES`. **Not yet called from the API path**, and **no garbage-input test yet**. |
| 4 | 🤖 3-tier escalation ladder | 🟡 Half done | Backend done: `ReviewTier` in schema, wired into `_to_prediction`. Threshold derived from cost ratio (0.0909 at 10:1). **Dashboard NOT updated** — still 2-state red/green. |
| 5 | 🤖 Calibration-drift monitor | ✅ Done | Two tiers on purpose: `quality.drift` (cheap KS for /api/health) + `mlops.drift` (PSI/cal/OOD for retrain verdict). Result: current batch flags DRIFT (6 of 7 features) — corroborates D-016 via a different statistic. |
| 6 | 🤖 Override audit trail | ✅ Verified | `OverrideRequest` already enforced user (min_length 1), reason (min_length 3), timestamp, target. No changes needed. |
| 7 | 🤖 Explanation-stability check | ⏳ Not started | One-off script perturbing rows, re-running SHAP, saving `explanation_stability.csv`. |
| 8 | 🤖 Fuzz-test the API | ⏳ Not started | Malformed dates, missing region_id, out-of-range lead_day → 4xx not 500. |
| 9 | 👤 Make one real override | ⏳ Waiting on you | One `curl POST /api/override` so `GET /api/overrides` shows something real. |
| 10 | 🤖 CI workflow | ✅ Done | `.github/workflows/ci.yml`, 5 jobs. Air-gap gate run locally, passes. Never run on Actions yet (no push). |
| 11 | 👤 Push to GitHub | ⏳ Waiting on you | Repo has no remote. |
| 12 | 👤 Create AWS account + $1 budget alarm | ⏳ Waiting on you | |
| 13 | 👤 Install/configure AWS CLI | ⏳ Waiting on you | This machine has NO AWS CLI, NO boto3, NO credentials (D-015). |
| 14 | 🤖+👤 S3 backup | ⏳ Waiting on you | Claude will give commands. |
| 15 | 👤 Docker Hub account | ⏳ Waiting on you | |
| 16 | 🤖+👤 Push image to Docker Hub | ⏳ Waiting on you | Local image `fbd:0.1.0` builds cleanly. |
| 17 | 👤 Launch EC2 t2.micro | ⏳ Waiting on you | |
| 18 | 👤 SSH + install Docker on EC2 | ⏳ Waiting on you | |
| 19 | 👤 Pull + run container on EC2 | ⏳ Waiting on you | |
| 20 | 👤 Verify public URL, paste into submission | ⏳ Waiting on you | Backup demo URL. |
| 21 | 👤 (optional) Bedrock model access | ⏳ Not started | GenAI code is written and unit-tested against a fake; no real call ever made. |
| 22 | 🤖 (optional) LLM narration into `bulletins.sqlite` | ⏳ Not started | **MUST run only in `generate_bulletins.py`**, never at request time. |
| 23 | 👤 (optional) NASA Earthdata login | ⏳ Not started | |
| 24 | 🤖 (optional) GPM IMERG cross-check | ⏳ Not started | |

### Rest of the SIH-strengthening roadmap (from `PROJECT_BLUEPRINT.md` §9.3)

| Item | Status | Notes |
|---|---|---|
| A. Multi-model AI ensembling (GraphCast/Pangu bust correlation) | ⏳ Not started | Highest-novelty remaining item. Would refresh D-014. |
| B. Retrospective case studies (Uttarakhand 2022, Chennai 2021, Wayanad 2024) | ⏳ Not started | Beat B in the strategy doc. |
| C. Uncertainty visualisation on the map | ⏳ Not started | Depends on Step 4b (dashboard tier state). |
| D. PDF bulletin export | ⏳ Not started | The paper artefact judges remember. |
| E. Cost-sensitivity slider | ⏳ Not started | Uses `metrics.value_at_cost_ratio`. |
| F. "Why refused?" panel for OOD subdivisions | ⏳ Not started | |
| G. Live IMD adapter (aspirational) | ⏳ Not started | IMD does not publish live NWP openly. |
| H. Mobile-responsive dashboard | ⏳ Not started | 1 hour. |
| I. Public hosted demo URL | ⏳ Same as walkthrough 17–20 | |
| J. 90-second demo script + rehearsal | ⏳ Not started | Not code work. |

---

## 8. Every user step, with what to do and what to expect

Ordered as you should actually do them. Skip freely.

### 8.1 Make one real override (Step 9, 30 seconds)

While the container is running (`docker compose up -d`):

```bash
curl -X POST http://localhost:8912/api/override \
  -H "Content-Type: application/json" \
  -d '{"region_id":"ASSAM_MEGHALAYA","init_date":"2022-06-14","lead_day":4,"action":"escalate","reason":"Confirmed via IMD Guwahati product; agreeing with the flag.","user":"harsh.trivedi"}'
```

**Expect:** `{"ok":true,"n_overrides":1,"note":"Override recorded. The system does not change any public warning."}`

Verify:
```bash
curl http://localhost:8912/api/overrides
```

**Note:** overrides live in the SQLite file inside the container. `docker compose down` and re-`up` will lose them unless you mount a volume. That is deliberate for a hackathon demo but worth knowing.

### 8.2 Push to GitHub (Step 11, 5 minutes)

1. Go to github.com, create a new **empty** repo (no README, no license). Suggested name: `forecast-bust-detection`.
2. In this repo:
   ```bash
   git remote add origin https://github.com/<your-username>/forecast-bust-detection.git
   git branch -M main
   git push -u origin main
   ```
3. Open the Actions tab on GitHub. **Expect:** a workflow run in progress, then a green checkmark within ~3 minutes.

If red: read the failure, share the log, ask Claude to fix.

### 8.3 AWS account + budget alarm (Step 12, 15 minutes)

1. aws.amazon.com → sign up (free tier, needs card).
2. **Immediately after:** Billing → Budgets → create budget → cost budget at **$1/month** with an email alert. Do this before anything else so you get warned if something starts costing money.
3. Console → IAM → Users → your user → Security credentials → Create access key → CLI. Save both keys.

### 8.4 Install AWS CLI (Step 13, 10 minutes)

Windows:
```powershell
winget install Amazon.AWSCLI
```
Then in bash:
```bash
aws configure
# paste the access key, secret, region "us-east-1", output "json"
aws sts get-caller-identity
```

**Expect:** JSON with your account id and ARN.

### 8.5 S3 backup (Step 14, 5 minutes)

```bash
BUCKET="fbd-sih26079-$(date +%s)"   # unique name
aws s3 mb "s3://$BUCKET" --region us-east-1
aws s3 cp data/artifacts/bulletins.sqlite "s3://$BUCKET/artifacts/"
aws s3 cp data/artifacts/bust_model.joblib "s3://$BUCKET/artifacts/"
aws s3 ls "s3://$BUCKET/artifacts/"
```

**Expect:** two lines listing the files with sizes.

### 8.6 Docker Hub + push image (Steps 15–16, 15 minutes)

1. hub.docker.com → sign up.
2. Locally:
   ```bash
   docker login
   docker tag fbd:0.1.0 <your-dockerhub-username>/fbd:0.1.0
   docker push <your-dockerhub-username>/fbd:0.1.0
   ```
3. Refresh your Docker Hub profile — the image should appear.

### 8.7 EC2 (Steps 17–20, 30 minutes)

1. Console → EC2 → Launch Instance:
   - Name: `fbd-demo`
   - AMI: **Amazon Linux 2023**
   - Instance type: **t2.micro** (free-tier)
   - Key pair: create a new one, download the `.pem` file, keep it safe
   - Network → Edit → **allow inbound TCP 8912 from 0.0.0.0/0** (security group)
   - Launch
2. Note the public IP. Then:
   ```bash
   chmod 400 ~/Downloads/your-key.pem
   ssh -i ~/Downloads/your-key.pem ec2-user@<public-ip>

   # inside EC2:
   sudo yum update -y
   sudo yum install -y docker
   sudo systemctl start docker
   sudo usermod -aG docker ec2-user
   exit
   # log back in
   ssh -i ~/Downloads/your-key.pem ec2-user@<public-ip>
   docker pull <your-dockerhub-username>/fbd:0.1.0
   docker run -d -p 8912:8912 --restart unless-stopped <your-dockerhub-username>/fbd:0.1.0
   ```
3. From your **laptop**:
   ```bash
   curl http://<public-ip>:8912/api/health
   ```
   **Expect:** 200 with `n_bulletin_rows: 79900`. Open `http://<public-ip>:8912` in a browser — the dashboard renders.

4. Paste the URL into the SIH submission form. **But do not demo against it.** The primary demo is your local container with wifi off. This is a backup link so judges can click something after the round.

### 8.8 (Optional) Bedrock access (Step 21)

Console → Bedrock → Model access → request `anthropic.claude-opus-5` and `anthropic.claude-haiku-4-5` in your chosen region. Approval is usually instant. Once approved, install `anthropic[bedrock]` and `boto3` and export `FBD_GENAI_ENABLED=1` — the code is already there (`src/fbd/genai/`).

---

## 9. Traps not to fall back into

Numbered as in `PROJECT_BLUEPRINT.md` §6, kept here for reference:

1. **WB2 Zarr chunks span the globe (D-002).** India subsetting saves memory and disk, not bandwidth.
2. **ERA5 store names lie (D-009).** The `1959-2022` store ends 2021-12-31. Use `1959-2023_01_10-6h-240x121` for anything touching 2022.
3. **IMD gridded rainfall is mainland-only (D-006).** A&N and Lakshadweep excluded; 34 subdivisions modelled.
4. **Bust definition needs all three clauses (D-004 / §4.3).** Percentile alone is circular; category alone misfires at 15.5 vs 15.7 mm.
5. **Fair calibration invariant (D-010).** Every baseline shares the model's isotonic on 2021.
6. **Day 10 lagged spread is 100% NaN (D-011).** Headline is Day 3–7 band.
7. **TreeSHAP via `shap` crashes on XGBoost 3.x (D-013).** Use `xgb.Booster.predict(..., pred_contribs=True)`.
8. **Season concatenation (D-011).** Group by year before `.diff()`.
9. **True ENS is a much stronger baseline than the lagged proxy (D-014).** The +0.082 headline was against the proxy. Real margin is +0.025 raw / +0.040 calibrated.
10. **`/api/bulletin/latest` is NOT a route.** Path param is `{region_id}`; `latest` returns 422.
11. **Non-ASCII in scripts breaks Windows cp1252 consoles.** ASCII-clean anything that prints.
12. ~~`git init` has not been run.~~ **DONE**, baseline commit `d3568ea`.
13. **LOGIC.md §13/§14 non-goals were amended (D-015).** GenAI/AWS is opt-in, default off, tested behind fakes. Never claim a Bedrock or cloud deployment as verified; the verification ledger in D-017 states what has actually been executed.
14. **The drift monitor's DRIFT verdict is a real finding, not a bug.** JJAS 2022 has genuinely lower shear than any other year in the record (D-016). Say it as "our held-out year is a harder test than assumed, and the model still scored 0.840".
15. **The GenAI numeric-grounding guardrail must never be relaxed.** An LLM sitting next to a disaster-management product cannot be allowed to state a probability the calibrated model did not compute.

---

## 10. Verification ledger (from D-017)

Say this whenever anyone asks what has actually been run vs. what is code-as-design:

| Component | Status |
|---|---|
| Full pipeline end-to-end | **Executed**, reproduces D-001..D-014 |
| 62 tests | **Executed**, 62/62 pass at `d66d188` |
| Drift monitors (both) | **Executed**; findings in D-016 |
| Docker + container health + dashboard air-gap | **Executed**, verified in a browser |
| GenAI guardrails/retrieval/tools/agent | **Executed** against a fake client |
| CI gates | **Executed locally**; never on GitHub Actions |
| Any real Bedrock call | **NEVER** — no SDK, no credentials |
| Terraform | **NEVER** — no binary, not even `validate`d |
| ECR push / ECS deploy / public URL | **NEVER** |
| Real override in the audit trail | **NEVER** — waiting on step 8.1 |

---

## 11. If everything else is lost — the shortest path to a live demo

```bash
cd C:/Users/ASUS/Desktop/sih
docker compose up -d
curl http://localhost:8912/api/health
start http://localhost:8912               # or: open http://localhost:8912
```

Assam & Meghalaya, 14 June 2022, Day 4: model flags at 70.6% while ensemble spread flags at 11.2%. That is your demo. Then say: *"It refuses on days it doesn't recognise. Those days bust at 23%, vs 3% for the rest. And it serves offline from a 63 MB SQLite file — the forecaster is always in the loop."*

That is enough. Everything else in this document exists to defend those two sentences.
