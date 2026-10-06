# Cloud and CI/CD plan

What exists, what is designed, and what needs your account or approval.
Written 2026-10-05 (master prompt §15–16), under three constraints:
- **No AWS spend.** The AWS path is withdrawn (D-021).
- **No billable service** is created without your explicit approval at the
  time.
- **No secrets** are asked for or stored by the assistant.

## 1. What exists today (implemented & verified)

### CI: `.github/workflows/ci.yml`, on every push and PR

| job | what it proves |
|---|---|
| `test` | the full pytest suite (330 tests on this branch) |
| `invariants` | decision-log invariants for the assistant layer: GenAI defaults to OFF; no write, override or alerting tool reaches the model; the numeric-grounding guardrail rejects an invented probability |
| `airgap` | The app imports with sockets blocked and GenAI routes stay unmounted. Every page's external origins are checked against an allowlist; imagery must default to disabled. |
| `security` | `pip-audit` on the serving requirements (**fails** on any finding not in the dated allowlist, C3), `bandit -r src/ -ll`, and a secret-pattern scan (fails) |
| `container` | a serving-layer image builds; `hadolint` on the Dockerfile |

### Release: `.github/workflows/publish.yml`, on `v*` tags

1. Build the image.
2. Load it locally and **smoke-test it**: the container must become
   healthy and serve a real bulletin.
3. Push to **GHCR** (`ghcr.io/<owner>/forecast-bust-detection`) with
   version tags.

Release `v0.2.0` carries `bulletins.sqlite`, checksum-pinned by
`scripts/fetch_release_artifacts.py`.

### Hosting (designed and documented, not live)

`docs/DEPLOY.md` (PR #5, green, open): **Render free tier** runs the GHCR
image read-only. With `FBD_READ_ONLY=1`, override returns 403 and GenAI is
off; about 100 MB of memory was measured. **Blocked on you:** a Render
account. Hugging Face Docker Spaces turned out to be paid.

## 2. Gaps, in priority order

| # | gap | plan | cost | needs you? |
|---|---|---|---|---|
| C1 | no live deployment | Create the Render web service from the GHCR image: port 8912, `FBD_READ_ONLY=1`, health check `/api/health`. Follow DEPLOY.md §3. | free tier | **yes**: account and clicks |
| C2 | the frontend has no visual regression in CI | Run `scripts/capture_screenshots.py` in CI (Chrome is on the runner) and assert `overflowX=false`, zero console errors and zero failed requests per page and viewport. Upload the PNGs as an artifact. No pixel diff yet, because SwiftShader output varies. | free | no |
| C3 | ~~`pip-audit` only warns~~ **done 2026-10-06** | Fail on high-severity findings in the serving set; keep an allowlist file with an expiry date per exception | free | no |
| C4 | no SBOM or provenance on images | Add `docker/build-push-action` `sbom: true` and `provenance: mode=max`; optionally cosign keyless signing with GitHub OIDC | free | no |
| C5 | ~~dependencies are unpinned by hash~~ **done 2026-10-06** (`requirements-serve.lock`, `--require-hashes`) | `pip-compile --generate-hashes` for `requirements-serve.txt`; install with `--require-hashes` in the image | free | no |
| C6 | no staged rollout | Use the `staging` → `latest` GHCR tags: staging deploys on merge to master, production on a `v*` tag. **Rollback** is redeploying the previous tag. | free | yes, for the second Render service |
| C7 | no scheduled integration check | Weekly `workflow_dispatch` plus cron: pull the latest GHCR image, run the smoke test, open an issue on failure | free | no |
| C8 | no ML pipeline CI | On PRs touching `src/fbd/model` or `evaluate`: run the registration-guard tests (`test_backtest_guard.py`, `test_promotion.py`) and a tiny-fixture training smoke test. **Never** a registered score in CI. | free | no |
| C9 | live ingestion (when built) | A cron job, a GitHub Action or a small always-on worker. Never inside the API process. Raw runs go to immutable storage. | the free GitHub Actions cron is enough for daily 00 UTC pulls of one source | yes, if cloud storage is wanted |

## 3. AWS (withdrawn, kept as a learning artifact)

The Terraform under `infra/` (VPC, ALB, Fargate, ECR) was **never applied**
and is labelled as such (D-017 → D-021). If funding returns, the master
prompt's §15 profile applies:
- a separate data plane (S3 raw/derived with versioning) from serving
- least-privilege IAM through GitHub OIDC (no long-lived keys)
- budgets and alarms first

**Do not** apply it, create keys, or open billable services without your
explicit approval at the time.

## 4. Definition of done for this plan

- **C1:** a public URL serving the read-only product, the health check
  green, the landing loading offline-capable assets only.
- **C2–C5, C7, C8:** each merged as its own PR with CI green.
- **C6:** two tags deployed, and one rollback exercised and recorded.
