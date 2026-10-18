# Assistant provider decision

**Date checked:** 2026-10-18. **Status:** default chosen for development and a
public demo; **not verified against a live call** (see "Verification boundary").

Provider pages change. Re-check every row below before a deployment, and update
the date. Model ids live in one place, `src/fbd/genai/research/settings.py`
(`PROVIDERS`), and each can be overridden with an environment variable, so a
renamed model is a configuration change, not a code change.

## Verification boundary

How each fact below was obtained, so nobody mistakes a summary for a check:

* The official pricing and limits pages could **not** be opened from the build
  container (its network policy does not resolve `ai.google.dev`,
  `tavily.com`, `openrouter.ai` or `console.groq.com`). The figures come from a
  web search on 2026-10-18, which returned the official OpenRouter limits page
  and the official Tavily credits documentation, and otherwise third-party
  summaries.
* Where search results disagreed, both readings are given and the row is
  marked **UNVERIFIED**.
* No real call to any of these providers has been made from this repository.
  The pipeline is tested end to end against mock providers (`tests/test_research_assistant.py`),
  which proves the wiring, validation and failure handling, not a vendor's
  behaviour. The ledger line below stays "never" until a key-backed run is
  recorded here.

| ledger | status |
|---|---|
| Real web search call | **never** (no key in the build environment; hosts unreachable) |
| Real hosted-LLM call | **never** (same) |
| Mocked end-to-end path | verified, 57 tests, every CI run |

## Search

| provider | free allowance | paid | notes | source of the figure | recommended use |
|---|---|---|---|---|---|
| **Tavily** (`FBD_SEARCH_PROVIDER=tavily`, `TAVILY_API_KEY`) | 1,000 credits / month, no card | pay-as-you-go $0.008 / credit; plans from $30 / month for 4,000 credits | basic search 1 credit, advanced 2; unused credits do not roll over (third-party) | official docs (docs.tavily.com, API credits) via search | **default** for development and demo. At the build's cap of 2 basic searches per question, the free month covers about 500 questions. |
| model-native grounding (e.g. Gemini "Grounding with Google Search") | reported as a monthly free allowance shared across models, then metered | reported at $14 per 1,000 searches for Gemini 3.x; older models priced differently | **UNVERIFIED**: sources disagree on the free allowance (5,000/month vs 500–1,500/day by model generation). Grounding returns redirect links, which cannot be checked against a source registry the way a search result's URL can. | third-party summaries | **not used.** The pipeline needs its own evidence packet with real URLs it can validate. |

## LLMs

| provider and model (default, overridable) | free tier | rate limits (free) | JSON / schema output | data use | source of the figure | recommended use |
|---|---|---|---|---|---|---|
| **Gemini** `gemini-flash-latest` (`GEMINI_API_KEY`, `FBD_GEMINI_MODEL`) | Flash and Flash-Lite models are reported free through AI Studio with no card; Pro models reported paid-only since April 2026 | per-model, per-project RPM/RPD; **UNVERIFIED** figures | `responseMimeType: application/json` | **On the free tier, Google states it may use prompts to improve its products; on paid usage it does not.** Do not send personal data on the free tier. | third-party summaries | **default** (free-first). The model alias is UNVERIFIED for this date: set `FBD_GEMINI_MODEL` to an id listed on the official models page. |
| **OpenRouter** `meta-llama/llama-3.3-70b-instruct:free` (`OPENROUTER_API_KEY`, `FBD_OPENROUTER_MODEL`) | `:free` variants | 20 requests/min; 50/day with under 10 credits purchased, 1,000/day after buying at least 10 credits | `response_format: json_object`, model-dependent | routed to third-party hosts; handling depends on the upstream provider | official limits page via search | **fallback** only, and only if configured. The `:free` model id is UNVERIFIED for this date. |
| **Groq** `llama-3.3-70b-versatile` (`GROQ_API_KEY`, `FBD_GROQ_MODEL`) | developer plan | about 30 RPM per model; daily request and token caps vary by model (one source: 1,000 req and 100K tokens/day for this model). **UNVERIFIED** | `response_format: json_object` | see Groq's terms before enabling | third-party summaries | **fallback** for low latency. Check the console Limits page for the account. |
| **Local** (Ollama) `llama3.1:8b` (`FBD_LOCAL_MODEL`) | no provider | hardware-bound; 15–35 s per answer on the 6 GB card measured in D-019 addendum 3 | Ollama `format: json` | nothing leaves the machine | measured in this repo (D-019) | **offline development**. D-019 measured 1-in-4 fabrication on narration at 8B, which is why every answer here is validated and row facts never pass through a model. |
| Paid options (Anthropic Claude, OpenAI GPT, Gemini Pro) | none | account-dependent | yes | paid tiers generally exclude training on prompts; check each | not checked this round | **premium opt-in** only: the operator must configure it, and `FBD_ASSISTANT_ALLOW_PAID=1` is required before any paid backend may serve as a fallback. Prices were not checked this round and are deliberately not quoted. |

## Decision

1. **Default: Gemini (free tier) for writing, Tavily (free tier) for search.**
   Both have a no-card free allowance large enough for development and a
   judged demo, and both are reached only from the server.
2. **Fallback: none by default.** An operator may set `FBD_ASSISTANT_FALLBACK`
   to `groq`, `openrouter` or `local`. A paid fallback is refused unless
   `FBD_ASSISTANT_ALLOW_PAID=1`. A fallback, when used, is announced on the
   answer ("written by the configured fallback model").
3. **Free tiers are for development and demos.** Their limits and data terms
   are not a basis for public traffic. For a public deployment, move to a paid
   tier whose terms exclude training on prompts, and keep the per-client and
   global caps (`FBD_RESEARCH_RPM`, `FBD_RESEARCH_RPD`, `FBD_RESEARCH_GLOBAL_RPD`).
4. **Evidence beats model size.** The model choice matters less than the
   retrieval and validation around it (D-019): a stronger model cannot make
   weak or missing evidence trustworthy, and the validator withholds an answer
   with an invented number whichever model wrote it.

## Evaluation still owed

`scripts/evaluate_assistant.py` runs the versioned question set in
`tests/assistant_eval/cases.jsonl`. With mocks it checks routing, abstention,
citation validity and safety handling (in CI). With keys it records, per model:
answer correctness against reviewed facts, citation validity and support rate,
unsupported-claim rate, abstention quality, schema-valid rate, p50/p95 latency
and token usage. **No model has been scored with keys yet.** Choose the
production default from that table, not from this one.

## Sources consulted (2026-10-18)

* Tavily API credits: https://docs.tavily.com/guides/api-credits
* OpenRouter limits: https://openrouter.ai/docs/api-reference/limits
* Gemini API pricing (official page to re-check): https://ai.google.dev/gemini-api/docs/pricing
* Groq models and limits (official pages to re-check): https://console.groq.com/docs/models , https://console.groq.com/docs/rate-limits
* Third-party summaries used where the official page was not reachable:
  https://www.cloudzero.com/blog/gemini-pricing/ ,
  https://www.aifreeapi.com/en/posts/google-gemini-api-free-tier ,
  https://www.firecrawl.dev/blog/tavily-pricing ,
  https://www.truefoundry.com/es/blog/openrouter-rate-limits-your-free-tier-depends-on-what-youve-already-paid ,
  https://eesel.ai/blog/groq-pricing , https://benchlm.ai/md/free-tier/groq.md
