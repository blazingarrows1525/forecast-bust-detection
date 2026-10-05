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

## 3. Free public URL — Render

The published image runs as-is on Render's free tier: no card, no build, no
data download. The whole API uses about 100 MB of memory (measured across the
dashboard, 3-D and metrics endpoints), well inside the free instance's 512 MB.

1. Sign up at <https://render.com>. Signing in with GitHub works, and the free
   tier needs no payment method.
2. **New → Web Service → Existing Image**. Image URL:
   `ghcr.io/blazingarrows1525/forecast-bust-detection:latest` (public, so no
   credential). Click **Connect**.
3. Name `forecast-bust-detection`, region **Singapore** (nearest to India),
   instance type **Free**.
4. Environment variables:
   * `PORT` = `8912`: the port the image serves on, so Render routes to it.
   * `FBD_READ_ONLY` = `1`: a public demo takes no anonymous writes to the
     forecaster audit log; `POST /api/override` answers 403 with a reason.
5. Under **Advanced**, set the health check path to `/api/health`, then
   **Deploy**. The URL is `https://forecast-bust-detection.onrender.com`, or
   that name with a suffix if it is taken.
6. When a new image is published (every push to `master`), use **Manual Deploy
   → Deploy latest reference** so Render pulls the new `:latest`.

`publish.yml` runs the image in read-only mode, as a different user (UID 1000),
before every push to GHCR. It refuses to publish unless the image serves and
the override returns 403.

**Limits to know:**
* A free service sleeps after 15 minutes without traffic, and the first
  request after that takes about a minute while it wakes. Open it before
  showing it to anyone.
* 750 free instance hours a month is enough for one service running all month.
* The filesystem is ephemeral, which does not matter for a read-only store.

### Other hosts, and why not

* **Hugging Face Spaces:** Docker Spaces now need a paid plan. The free Gradio
  and Static tiers cannot run this container.
* **Google Cloud Run:** free at this volume, but needs a billing account on
  file:

```bash
gcloud run deploy fbd   --image ghcr.io/blazingarrows1525/forecast-bust-detection:latest   --region asia-south1 --allow-unauthenticated   --memory 512Mi --cpu 1 --min-instances 0 --max-instances 2   --port 8912 --set-env-vars FBD_READ_ONLY=1
```

* **Fly.io:** needs a card for new accounts.

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
| Hosting | Render free web service | ₹0 |
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
