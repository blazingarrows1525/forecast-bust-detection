# DEPLOY.md — running this system for zero rupees

Every option here is free at the volume this project needs. Nothing requires a
card on file. The AWS path is withdrawn, not deferred — see `DECISIONS.md`
D-021.

---

## 0. What actually has to run

The serving layer is deliberately small, and that is what makes free hosting
viable rather than a compromise:

* one FastAPI process,
* one ~74 MB SQLite file of precomputed bulletins,
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

Best fit: it is ML-native, reviewers recognise it, Docker Spaces are free with
no card, and the URL is stable. The Space does not build this repo. It runs
the published image through the two files in
[`deploy/huggingface/`](../deploy/huggingface/). Pushing this repo to a Space
would build an image without the bulletin store, which is a Release asset and
not in git (`DATA.md`), so every panel would be empty.

1. Create a free account at <https://huggingface.co/join>.
2. Create a Space at <https://huggingface.co/new-space>: name
   `forecast-bust-detection`, SDK **Docker** (blank template), hardware
   **CPU basic** (free), visibility **Public**.
3. Put the two files in it. The simplest way is in the browser: **Files → Add
   file → Upload files**, then upload `deploy/huggingface/Dockerfile` and
   `deploy/huggingface/README.md` (replacing the generated README) and commit.
   Or with git (git asks for your username and an access token with write
   access):

```bash
git clone https://huggingface.co/spaces/<your-username>/forecast-bust-detection hf-space
cp deploy/huggingface/Dockerfile deploy/huggingface/README.md hf-space/
cd hf-space && git add . && git commit -m "Run the published image" && git push
```

4. The Space pulls the image and starts in a few minutes. The URL is
   `https://<your-username>-forecast-bust-detection.hf.space`.
5. When a new image is published (every push to `master`), open the Space's
   **Settings → Factory rebuild** so it pulls the new `:latest`.

**Read-only by design.** The Space sets `FBD_READ_ONLY=1`, so
`POST /api/override` answers 403 with a message instead of writing
anonymous entries into the forecaster audit log. A Space also runs the
container as UID 1000, which cannot write the image's data directory.
`publish.yml` runs the image exactly that way, as UID 1000 in read-only mode,
before every push to GHCR, and refuses to publish if it does not serve.

**Limit to know:** free Spaces sleep when idle and cold-start in about 30 s.
Wake it before showing it to anyone.

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

Both have free allowances that fit, and both can run the published image
directly. Point the service at
`ghcr.io/blazingarrows1525/forecast-bust-detection:latest`, set port 8912, and
set `FBD_READ_ONLY=1` for a public instance. Render free web services sleep
when idle, with the same caveat as Spaces.

---

## 4. The assistant layer (optional, off by default)

The GenAI assistant is opt-in and defaults to a **local** model, so it needs no
account and no spend (D-019).

```bash
# one-time
ollama pull llama3.1:8b

# run with the assistant enabled
FBD_GENAI_ENABLED=1 FBD_GENAI_TOOLS=1 FBD_GENAI_RAG=1 \
PYTHONPATH=src python -m uvicorn fbd.api.app:app --port 8912
```

Configuration:

| variable | default | meaning |
|---|---|---|
| `FBD_GENAI_ENABLED` | off | master switch; off means no routes are registered at all |
| `FBD_GENAI_PROVIDER` | `local` | `local` (Ollama) or `bedrock` (managed cloud) |
| `FBD_LOCAL_MODEL` | `llama3.1:8b` | any pulled Ollama tag (~5 GB VRAM at 8B; `llama3.2:3b` fits 2 GB but fabricates far more -- see D-019 addendum 3) |
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
