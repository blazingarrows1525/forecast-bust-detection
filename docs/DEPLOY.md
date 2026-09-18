# DEPLOY.md — running this system for zero rupees

Every option here is free at the volume this project needs. Nothing requires a
card on file. The AWS path is withdrawn, not deferred — see `DECISIONS.md`
D-021.

---

## 0. What actually has to run

The serving layer is deliberately small, and that is what makes free hosting
viable rather than a compromise:

* one FastAPI process,
* one ~60 MB SQLite file of precomputed bulletins,
* two precomputed geo assets (0.64 MB),
* vendored frontend assets (no CDN).

**No model runs at request time.** No raw NWP data, no GPU, no geo stack
(D-020), no outbound network. The daily batch that produces the bulletins runs
offline on a laptop in ~4 minutes and is not part of the deployment.

That is why this fits inside free tiers with room to spare: the container is a
static-ish read path, and it scales to zero between requests.

---

## 1. Local — the one that always works

```bash
docker compose up -d
curl http://localhost:8912/api/health
```

Dashboards: `http://localhost:8912/` (2D map) and `/command.html` (3D risk cube).

**This is the demo path.** It needs no internet once built, which is the
property the whole system is designed around. If venue wifi fails, this still
runs.

---

## 2. Pull the published image

Built and smoke-tested by `.github/workflows/publish.yml` on every push to
`master`, then pushed to GHCR. Free for public repositories — Actions minutes
are unmetered and public package storage is free.

```bash
docker run -p 8912:8912 ghcr.io/blazingarrows1525/forecast-bust-detection:latest
```

The workflow refuses to publish an image that fails a boot-and-serve smoke
test, and fails the build if `geopandas` is importable inside it — so the D-020
optimisation cannot silently regress.

---

## 3. Free public URL — pick one

Ranked by fit for this project.

### 3a. Hugging Face Spaces — recommended

Best fit: it is ML-native, judges and reviewers recognise it, Docker Spaces are
free and persistent, and the URL is stable.

1. Create a Space → SDK **Docker** → visibility Public.
2. Push this repo to the Space remote (it builds the `Dockerfile` directly).
3. HF serves on port 7860, so either set `app_port: 8912` in the Space README
   front-matter, or override the command:

```yaml
---
title: Forecast Bust Detection
sdk: docker
app_port: 8912
---
```

**Limit to know:** free Spaces sleep after inactivity and cold-start in ~30 s.
Wake it before a demo.

### 3b. Google Cloud Run

The most production-shaped option. Free tier is 2 M requests, 360 k GB-seconds
and 180 k vCPU-seconds per month — orders of magnitude above this project's
usage. Scales to zero, so idle costs nothing.

```bash
gcloud run deploy fbd \
  --image ghcr.io/blazingarrows1525/forecast-bust-detection:latest \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 2 \
  --port 8912
```

`--min-instances 0` is the cost control: no idle billing, at the price of a
cold start. `--max-instances 2` caps the blast radius if something loops.
`asia-south1` (Mumbai) is nearest to the users this system is for.

**Requires a billing account on file**, even though the free tier covers this
workload. If that is not acceptable, use Spaces.

### 3c. Fly.io / Render

Both have free allowances that fit. Render free web services sleep on
inactivity, same caveat as Spaces.

---

## 4. The assistant layer (optional, off by default)

The GenAI assistant is opt-in and defaults to a **local** model, so it needs no
account and no spend (D-019).

```bash
# one-time
ollama pull llama3.2:3b

# run with the assistant enabled
FBD_GENAI_ENABLED=1 FBD_GENAI_TOOLS=1 FBD_GENAI_RAG=1 \
PYTHONPATH=src python -m uvicorn fbd.api.app:app --port 8912
```

Configuration:

| variable | default | meaning |
|---|---|---|
| `FBD_GENAI_ENABLED` | off | master switch; off means no routes are registered at all |
| `FBD_GENAI_PROVIDER` | `local` | `local` (Ollama) or `bedrock` (managed cloud) |
| `FBD_LOCAL_MODEL` | `llama3.2:3b` | any pulled Ollama tag |
| `FBD_OLLAMA_HOST` | `http://localhost:11434` | where Ollama listens |

`/api/health` reports which provider is configured and, if it cannot run, the
specific reason — so an operator never has to guess whether the assistant is
live.

**The assistant is not on the demo path.** The dashboards, the bulletins and
every number in the evaluation work with it switched off.

---

## 5. Cost summary

| component | service | monthly cost |
|---|---|---|
| CI | GitHub Actions (public repo) | ₹0 — unmetered |
| Registry | GHCR (public package) | ₹0 |
| Hosting | HF Spaces or Cloud Run free tier | ₹0 |
| LLM | Ollama, local | ₹0 |
| Data | IMD + WeatherBench 2, public archives | ₹0 |
| **Total** | | **₹0** |

---

## 6. What is deliberately not here

* **Kubernetes, service mesh, autoscaling groups.** This is a daily batch job
  serving a static artifact. `LOGIC.md` §14 rules them out and nothing since
  has changed that.
* **A managed database.** The bulletin store is read-only at request time;
  SQLite in the image is the correct shape, and it is what makes the offline
  demo possible.
* **Real-time streaming.** The forecast cycle is once daily.

Each of these is a non-goal with a reason, not an omission.
