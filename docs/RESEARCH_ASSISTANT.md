# Research assistant

An evidence-grounded assistant for questions about a reading, the method and,
when an operator enables it, current public information. It lives at
`/assistant.html` ("Ask" in every view's navigation).

It summarises sources and the archive. **It is not a warning service**, it
cannot send or publish anything, and every response carries
`authoritative: false`.

## Status

| part | status |
|---|---|
| Routing, archive answers, document answers, SSRF-safe fetching, citation and number validation, quotas, SSE streaming, the page | implemented and tested against mocks (57 tests in `tests/test_research_assistant.py` and `tests/test_assistant_eval.py`, run in CI) |
| Archive and document answers against the real bulletin store | verified in a browser in this build, with the server started as in "Run it" below |
| Real web search (Tavily) and hosted models (Gemini, OpenRouter, Groq) | **UNVERIFIED**: no key in the build environment and the hosts were unreachable from it. Wired, tested with mocks, never called for real. |

## How it answers

```
question ─► router (fixed rules, no model)
              │
              ├─ a reading attached, "why / explain / compare"  ─► archive row ─► written from a template
              │                                                    (no model; D-019)
              ├─ method, definitions, how-to                     ─► project documents (BM25) ─► model, validated
              │                                                    or excerpts if no model is configured
              ├─ current information, advisories                 ─► web search ─► safe fetch ─► model, validated
              ├─ "send / issue an alert"                         ─► "read-only; here is who issues warnings"
              └─ unclear, or a place is needed                   ─► one question back
```

Rules that hold on every path:

* **Facts about a reading are read, never written.** Status, probability,
  tier, reasons and forecast-minus-observed arithmetic come from
  `bulletins.sqlite` and a template. No model sees them. This is the D-019
  finding applied: a model narrated a scored row as "refused" three times out
  of three.
* **Every claim cites a source the server retrieved for this request.** Source
  ids are issued per request (`S1`, `S2`, …). The validator removes any other
  id, and the page renders citations only for ids it was sent. The page never
  turns model text into a link.
* **No number the sources do not contain.** The existing numeric-grounding
  guardrail runs on the answer against the numbers in the cited sources. One
  invented number withholds the whole answer; the user sees the sources and
  "Unable to verify".
* **No directives.** The non-interference guardrail (LOGIC.md 1.3) runs on
  every answer. "Residents should evacuate" is withheld even when a source says
  it; the answer points to the official agency instead.
* **Web pages are data.** Each page is screened for instruction-shaped text;
  a flagged page is shown to the user with the flag and withheld from the
  model. The packet fences every source between untrusted-data markers.
* **No answer from memory** for current questions. No sources means
  "Unable to verify", not a guess.

## Run it

The assistant is off by default and the routes do not exist until enabled.

```bash
# the Claude Code cloud environment -- keys in Network secrets, nothing in code
FBD_GENAI_ENABLED=1 FBD_RESEARCH=1 FBD_RESEARCH_WEB=1 \
  FBD_ASSISTANT_LLM=gemini FBD_ASSISTANT_FALLBACK=openrouter FBD_ASSISTANT_PROXY_AUTH=1 \
  PYTHONPATH=src python -m uvicorn fbd.api.app:app --app-dir src --port 8912

# archive and document answers only: no key, no network, no model
FBD_GENAI_ENABLED=1 FBD_RESEARCH=1 FBD_ASSISTANT_LLM=none \
  PYTHONPATH=src python -m uvicorn fbd.api.app:app --app-dir src --port 8912

# with a free-tier model for document synthesis
FBD_GENAI_ENABLED=1 FBD_RESEARCH=1 FBD_ASSISTANT_LLM=gemini GEMINI_API_KEY=... \
  PYTHONPATH=src python -m uvicorn fbd.api.app:app --app-dir src --port 8912

# plus web search
FBD_GENAI_ENABLED=1 FBD_RESEARCH=1 FBD_RESEARCH_WEB=1 \
  FBD_ASSISTANT_LLM=gemini GEMINI_API_KEY=... TAVILY_API_KEY=... \
  PYTHONPATH=src python -m uvicorn fbd.api.app:app --app-dir src --port 8912
```

Keys are read from the server's environment only. In a cloud environment, add
them as secrets (never in the page, a URL or a commit). `/api/assistant/research/status`
says whether each key is set and never prints one.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `FBD_GENAI_ENABLED` | off | master switch for the whole GenAI layer (D-015) |
| `FBD_RESEARCH` | off | mounts the research assistant's routes |
| `FBD_RESEARCH_WEB` | off | allows outbound web search and page fetches |
| `FBD_ASSISTANT_LLM` | `gemini` | `gemini`, `openrouter`, `groq`, `local` (Ollama), or `none` |
| `FBD_ASSISTANT_FALLBACK` | empty | a second backend, used only if set; announced on the answer |
| `FBD_ASSISTANT_ALLOW_PAID` | off | required before a paid backend may be a fallback |
| `FBD_SEARCH_PROVIDER` | `tavily` | `tavily` or `mock` |
| `GEMINI_API_KEY`, `TAVILY_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY` | none | provider keys |
| `FBD_GEMINI_MODEL`, `FBD_OPENROUTER_MODEL`, `FBD_GROQ_MODEL`, `FBD_LOCAL_MODEL` | see `settings.PROVIDERS` | model ids; check them against the provider before deploying |
| `FBD_RESEARCH_MAX_SEARCHES` | 2 | searches per question |
| `FBD_RESEARCH_MAX_FETCHES` | 3 | pages read in full per question |
| `FBD_RESEARCH_MAX_DOCS` | 6 | sources kept per answer |
| `FBD_RESEARCH_MAX_PAGE_BYTES` / `_CHARS` | 1.5 MB / 5,000 | read and keep caps per page |
| `FBD_RESEARCH_MAX_OUTPUT_TOKENS` | 1,200 | model output cap |
| `FBD_RESEARCH_DEADLINE_S` | 45 | wall-clock budget per question |
| `FBD_RESEARCH_RPM` / `_RPD` | 6 / 60 | per-client limits |
| `FBD_RESEARCH_GLOBAL_RPD` | 400 | global daily cap |
| `FBD_RESEARCH_ALLOWED_ORIGINS` | empty | other web origins allowed to call chat; by default only the site's own pages may (403 otherwise) |
| `FBD_ASSISTANT_PROXY_AUTH` | off | **set to `1` when the keys live in Claude Code's Network secrets** (or any outbound proxy that injects auth headers). The code then neither checks for a local key nor sends an auth header; the proxy adds the right one for the host, and the key never reaches this process. |

Provider choice, free-tier limits and data terms: `docs/assistant-provider-decision.md`.

## API

`POST /api/assistant/research/chat`

```json
{"message": "Why was this flagged?",
 "conversation_id": "optional-8-to-64-chars",
 "context_refs": [{"region_id": "ASSAM_MEGHALAYA", "init_date": "2022-06-14", "lead_day": 4}],
 "web_mode": "auto | on | off",
 "response_mode": "stream | complete"}
```

Streaming is server-sent events, each with an increasing `id`:

| event | data |
|---|---|
| `status` | `{stage, label, request_id}`: routing, searching, reading, checking, writing |
| `source` | a source's public fields (title, publisher, published/retrieved time, trust tier, fetch status, flags) |
| `warning` | `{code, message}`, e.g. a fallback model was used, or an answer failed verification |
| `done` | `{answer}`: the full `AssistantAnswer` (schemas.py) |
| `error` | `{code, message, retryable}`: sanitised, never a stack trace |

There are **no token events**. The model is called in JSON mode and the answer
is validated as a whole before it is sent; streaming prose that might then be
withheld would show the user exactly what the checks exist to stop. Stages
stream in real time; the answer arrives once. The page drops any event whose id
it has already seen, and a dropped connection shows "Interrupted" rather than a
partial answer.

**Cancellation.** Stop in the page aborts the request. The server stops
iterating the pipeline; a provider call already in flight finishes in its
worker thread and is discarded, and no further search, fetch or model call is
made. (urllib cannot abort a request mid-flight; this is the honest limit.)

## Costs and abuse

* Per-client and global quotas (429 with `Retry-After`).
* Chat answers only the site's own pages: a browser request from another origin gets 403, so
  another website cannot make its visitors spend this deployment's budget. (The read-only data
  endpoints keep their open CORS policy.)
* At most 2 searches, 3 page reads and 1,200 output tokens per question, with a
  45 s deadline.
* Search results are cached by query, place, time scope, provider and config
  version: 5 minutes for current and safety questions, 6 hours otherwise. Every
  answer shows each source's retrieval time, so a cached advisory is never shown
  as fresher than it is.
* At the defaults, Tavily's free month (1,000 credits) covers about 500
  questions that search.

## Privacy

* The server keeps the last question per conversation id for 30 minutes, in
  memory, only to resolve "Which place?" follow-ups. Nothing is written to disk.
* Logs record the request id, route, status, counts and latency, plus a hash
  and the length of the question. They do not record its text.
* The browser's location is never requested. A question that needs a place
  asks for one.
* On free tiers, providers may use prompts to improve their products (see the
  decision record). Do not enable a free-tier model for questions that carry
  personal data.

## Tests and evaluation

```bash
PYTHONPATH=src python -m pytest tests/test_research_assistant.py tests/test_assistant_eval.py -q
python scripts/evaluate_assistant.py            # 19 cases, mocks, what CI runs
python scripts/evaluate_assistant.py --live     # the configured providers; writes artifacts/eval/live.json
```

The case file `tests/assistant_eval/cases.jsonl` covers safety questions with
and without a place, current questions with no results and with web off, an
invented number, an invented citation, an injected page, a metadata-endpoint
URL in the results, a request to send an alert, an ambiguous message, and
archive explanations, including a refused row that must never be called low.
**Add a case for every failure found in use.**

## Troubleshooting

| symptom | cause |
|---|---|
| The page says "off in this build" | `FBD_GENAI_ENABLED` or `FBD_RESEARCH` is not set; the routes do not exist |
| "Web search is off in this build" | `FBD_RESEARCH_WEB` is not set, or the page's Web search menu is Off |
| Status shows `key_configured: false` | the key variable is not set on the server process |
| `circuit_open` errors | three consecutive provider failures; the provider is skipped for 60 s |
| Answers are excerpts, not prose | no model configured (`FBD_ASSISTANT_LLM=none`), or the model failed and no fallback is set |
| "did not pass verification" | the model wrote a number its sources do not contain, or a directive; read the sources directly |
| 429 | per-client or global quota; see `Retry-After` |
