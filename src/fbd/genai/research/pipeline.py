"""The research pipeline: route, retrieve, check, write, validate -- in that order.

``run()`` is a generator of events, so the API can stream real stages to the
page and the evaluation harness can consume the same generator directly:

    status   {stage, label, request_id}       a stage actually started
    source   {SourceDocument, public fields}  a source was registered
    warning  {code, message}                  a limitation the user should see
    done     {answer: AssistantAnswer}        the final, validated result
    error    {code, message, retryable}       nothing usable could be produced

There are no "token" events. The model is called in JSON mode and validated as
a whole before anything is shown; streaming unvalidated prose to the page and
then retracting it would show the user exactly the text the checks exist to
stop. Stages stream; the answer arrives once, checked.

What never goes through a model (D-019): a row's status, probability, tier and
reasons, and any forecast/observation arithmetic. Those are read from the
archive and written from templates. The model is used only to synthesise web
or documentation sources, and its output is validated (validate.py).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterator

from fbd import config
from fbd.genai.research import evidence as ev
from fbd.genai.research import extract, prompt, router, urlsafe, validate
from fbd.genai.research.providers import ProviderFailure, build_llm, build_search
from fbd.genai.research.schemas import (
    AssistantAnswer, ChatRequest, ClaimEvidence, Intent, RouteDecision, SourceDocument, TrustTier,
)
from fbd.genai.research.settings import ResearchSettings
from fbd.genai.research.state import MEMORY, SEARCH_CACHE
from fbd.obs.metrics import REGISTRY, get_logger, log_event

LOG = get_logger("fbd.research")
CONFIG_VERSION = "1"

SAFETY_NOTE = (
    "This assistant summarises sources and the archive; it is not a warning service. For warnings "
    "and protective action, follow the India Meteorological Department and your State or District "
    "Disaster Management Authority.")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Deps:
    """Everything with a side effect, injectable for tests and evals."""

    search: object | None = None
    llm: object | None = None
    fallback_llm: object | None = None
    fetch: Callable[[str], dict] | None = None   # url -> {"url","title","text","published_at","site_name"}
    row: Callable[[str, str, int], dict | None] | None = None
    docs: Callable[[str, int], list[dict]] | None = None
    clock: Callable[[], str] = _now
    fallback_name: str | None = None


@dataclass
class Run:
    req: ChatRequest
    settings: ResearchSettings
    deps: Deps
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    t0: float = field(default_factory=time.monotonic)
    sources: list[SourceDocument] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def deadline(self) -> float:
        return self.t0 + self.settings.deadline_s

    def expired(self) -> bool:
        return time.monotonic() > self.deadline

    def add(self, doc: SourceDocument) -> SourceDocument:
        doc.source_id = f"S{len(self.sources) + 1}"
        self.sources.append(doc)
        return doc


# ----------------------------------------------------------- default deps
def default_row(region_id: str, init_date: str, lead: int) -> dict | None:
    db = config.ARTIFACTS / "bulletins.sqlite"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT * FROM bulletins WHERE region_id = ? AND init_date = ? AND lead_day = ?",
                        (region_id, init_date, int(lead))).fetchone()
    finally:
        con.close()
    if r is None:
        return None
    row = {k: r[k] for k in r.keys()}
    try:
        row["dominant_factors"] = json.loads(row.get("dominant_factors") or "[]")
    except (TypeError, ValueError):
        row["dominant_factors"] = []
    from fbd.quality import escalation
    tier = escalation.classify_tier(bust_probability=row.get("bust_probability"),
                                    ood_flag=(row.get("status") != "OK"))
    row["review_tier"], row["tier_guidance"] = tier.value, escalation.guidance(tier)
    return row


_DOC_FILES = ("README.md", "FRONTEND_LOGIC.md", "LOGIC.md", "DECISIONS.md", "docs/PROJECT_GUIDE.md", "PRODUCT.md",
              "docs/USING_THE_VIEWS.md", "docs/RESEARCH_ASSISTANT.md")
_doc_index = None


# BM25 matches words, and people ask in words the documents do not use. A
# small, inspectable table maps the product's own vocabulary onto the terms its
# documents are written in; it adds terms, it never removes the user's.
_EXPAND = (
    (r"refus|declin|out[- ]of[- ]distribution|\bood\b|not scored", "refusal refused out of distribution unknown risk"),
    (r"bust", "bust definition magnitude category flip operational significance"),
    (r"tier|review|flag", "review tier AUTO_OK REVIEW REFUSE"),
    (r"lead|timeline|play|scrub|shortcut|keyboard", "lead timeline playback keyboard shortcuts views"),
    (r"baseline|spread|ensemble", "ensemble spread baseline combination"),
    (r"calibrat|reliab", "calibration isotonic reliability"),
    (r"why .*number|provenance|where .* come from", "provenance why this number source"),
)


def expand(query: str) -> str:
    import re
    extra = [terms for pat, terms in _EXPAND if re.search(pat, query, re.IGNORECASE)]
    return query + (" " + " ".join(extra) * 2 if extra else "")


def default_docs(query: str, k: int = 3) -> list[dict]:
    """BM25 over the project's own documents (retrieval.py), including the UI docs."""
    global _doc_index
    from fbd.genai import retrieval
    if _doc_index is None:
        passages = []
        for name in _DOC_FILES:
            path = config.ROOT / name
            if path.exists():
                passages.extend(retrieval._split_sections(name, path.read_text(encoding="utf-8")))
        _doc_index = retrieval.DocIndex(passages)
    return [{"doc": p.doc, "heading": p.heading, "text": p.text, "score": s}
            for p, s in _doc_index.search(expand(query), k=k)]


def default_fetch(url: str, settings: ResearchSettings) -> dict:
    page = urlsafe.fetch(url, max_bytes=settings.max_page_bytes)
    if page.status != 200:
        raise OSError(f"HTTP {page.status}")
    out = extract.extract(page.text, page.content_type, settings.max_page_chars)
    out["url"] = page.url
    return out


def default_deps(settings: ResearchSettings) -> Deps:
    fb = settings.fallback_spec()
    return Deps(
        search=build_search(settings.search_provider) if settings.web else None,
        llm=build_llm(settings.spec()),
        fallback_llm=build_llm(fb), fallback_name=fb.name if fb else None,
        fetch=lambda u: default_fetch(u, settings), row=default_row, docs=default_docs,
    )


# ---------------------------------------------------------------- helpers
def _ev(kind: str, **data) -> dict:
    return {"event": kind, "data": data}


def _public(doc: SourceDocument) -> dict:
    return doc.model_dump(mode="json")


def _pct(x) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "no value"


def _host(url: str) -> str:
    import urllib.parse
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _row_source(run: Run, ref, row: dict) -> SourceDocument:
    text = json.dumps({k: row.get(k) for k in (
        "region", "region_id", "init_date", "valid_date", "lead_day", "status", "bust_probability",
        "model_probability", "baseline_probability", "ens_spread", "forecast_rain_mm", "observed_rain_mm",
        "actual_bust", "review_tier", "tier_guidance", "dominant_factors", "model_version")}, default=str)
    return run.add(SourceDocument(
        source_id="", canonical_url=f"/api/bulletin/{ref.region_id}?init_date={ref.init_date}&mode=replay",
        title=f"Archive record: {row.get('region')}, issued {ref.init_date}, Day {ref.lead_day}",
        publisher="Forecast Bust Intelligence archive (bulletins.sqlite)", published_at=None,
        retrieved_at=run.deps.clock(), content_hash=ev.content_hash(text), extracted_text=text,
        excerpt=f"status {row.get('status')}, served {_pct(row.get('bust_probability'))}",
        trust_tier=TrustTier.PROJECT, fetch_status="internal"))


def _prose(text: str, n: int) -> str:
    """A readable excerpt of a markdown section: no tables or code fences, cut
    at a word boundary. The model still receives the full section."""
    keep, fence = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence or s.startswith("|") or not s:
            continue
        keep.append(s)
    flat = " ".join(keep)
    if len(flat) <= n:
        return flat
    cut = flat.rfind(" ", 0, n)
    return flat[: cut if cut > n * 0.6 else n].rstrip(",;:") + "…"


def _doc_sources(run: Run, query: str, k: int = 2) -> list[SourceDocument]:
    out = []
    for hit in (run.deps.docs(query, k) if run.deps.docs else []):
        text = hit["text"][:3000]
        doc = ev.screen(SourceDocument(
            source_id="", canonical_url=f"repo:{hit['doc']}#{hit['heading']}",
            title=f"{hit['doc']}: {hit['heading'] or 'introduction'}", publisher="Project documentation",
            published_at=None, retrieved_at=run.deps.clock(), content_hash=ev.content_hash(text),
            extracted_text=text, excerpt=_prose(text, 600), trust_tier=TrustTier.PROJECT, fetch_status="internal"))
        out.append(run.add(doc))
    return out


def _answer(run: Run, route: RouteDecision, markdown: str, status: str, *, claims=None, key_points=None,
            limitations=None, llm=None, fallback=False, flags=None, clarify=False) -> AssistantAnswer:
    cited = set(validate.CITE_RE.findall(markdown)) | {s for c in (claims or []) for s in c.source_ids}
    # The source list shows what was cited first, then what was retrieved and not cited.
    ordered = sorted(run.sources, key=lambda d: (d.source_id not in cited, int(d.source_id[1:])))
    safety = SAFETY_NOTE if route.safety_level != "none" else None
    return AssistantAnswer(
        answer_markdown=markdown, key_points=key_points or [], citations=ordered, claims=claims or [],
        limitations=(limitations or []) + run.warnings, retrieval_status=status, route=route,
        searched=run.searched, model_id=getattr(llm, "model", None), provider_id=getattr(llm, "provider", None),
        fallback_used=fallback, created_at=run.deps.clock(),
        latency_ms=int((time.monotonic() - run.t0) * 1000), usage=getattr(llm, "usage", {}) or {},
        safety_flags=flags or [], safety_note=safety, needs_clarification=clarify)


# --------------------------------------------------------- deterministic
def _explain_row(run: Run, route: RouteDecision) -> Iterator[dict]:
    ref = run.req.context_refs[0]
    yield _ev("status", stage="reading", label="Reading the archive record", request_id=run.request_id)
    row = run.deps.row(ref.region_id, ref.init_date, ref.lead_day) if run.deps.row else None
    if not row:
        yield _ev("done", answer=_answer(run, route, (
            f"There is no archive record for {ref.region_id} issued {ref.init_date} at Day {ref.lead_day}, "
            "so there is nothing to explain. Check the date and lead on the map."), "unable_to_verify").model_dump(mode="json"))
        return
    src = _row_source(run, ref, row)
    yield _ev("source", source=_public(src))
    s = src.source_id
    name, L = row.get("region"), ref.lead_day
    lines: list[str] = []
    claims: list[ClaimEvidence] = []

    if route.intent == Intent.DATA_COMPARISON:
        f, o = row.get("forecast_rain_mm"), row.get("observed_rain_mm")
        if isinstance(f, (int, float)) and isinstance(o, (int, float)):
            diff = f - o
            lines.append(f"**{name}, Day {L}** (issued {ref.init_date}, valid {row.get('valid_date')}): the forecast "
                         f"was **{f:.1f} mm/day** and **{o:.1f} mm/day** was observed, a difference of "
                         f"**{diff:+.1f} mm/day** (forecast minus observed) [{s}].")
        else:
            lines.append(f"The archive has no observation for {name} at Day {L}, so the forecast cannot be "
                         f"compared with what fell [{s}]. That is a missing value, not zero.")
        b = row.get("actual_bust")
        lines.append({1: f"Under the project's definition, this forecast **busted** [{s}].",
                      0: f"Under the project's definition, this forecast **held** [{s}]."}.get(
            b, f"Whether it busted is not recorded [{s}]."))
        claims.append(ClaimEvidence(claim_id="C1", claim_text="forecast and observed rainfall", source_ids=[s]))
        defn = _doc_sources(run, "bust label definition magnitude category flip operational significance", 1)
        for d in defn:
            yield _ev("source", source=_public(d))
            lines.append(f"A bust needs all three at once: a large error, a change of IMD intensity band, and "
                         f"an operationally significant amount on at least one side; the thresholds are in "
                         f"the definition [{d.source_id}].")
    else:
        if row.get("status") != "OK":
            lines.append(f"**{name}, Day {L}** (issued {ref.init_date}): the system **declined to score** this "
                         f"row (status `{row.get('status')}`) [{s}]. That is unknown, elevated risk, not a low "
                         "reading: the atmospheric state was unlike anything in training.")
        else:
            lines.append(f"**{name}, Day {L}** (issued {ref.init_date}, valid {row.get('valid_date')}): the served "
                         f"probability that this forecast busts is **{_pct(row.get('bust_probability'))}**; the "
                         f"ensemble-spread baseline gave **{_pct(row.get('baseline_probability'))}** [{s}].")
            if isinstance(row.get("model_probability"), (int, float)):
                spread = row.get("ens_spread")
                spread_txt = f"spread {spread:.1f} mm/day" if isinstance(spread, (int, float)) else "spread not recorded"
                lines.append(f"The served number combines the model ({_pct(row['model_probability'])}) with the "
                             f"50-member ensemble ({spread_txt}) [{s}].")
        lines.append(f"Review tier: **{row.get('review_tier')}**. {row.get('tier_guidance') or ''} [{s}]")
        reasons = row.get("dominant_factors") or []
        if reasons:
            lines.append("Reasons the system gives, verbatim:")
            lines.extend(f"- {r} [{s}]" for r in reasons[:5])
        claims.append(ClaimEvidence(claim_id="C1", claim_text="status, probability and tier of the record",
                                    source_ids=[s]))
    if route.intent != Intent.DATA_COMPARISON:
        # What the tier and a refusal mean, from a fixed section of the
        # project's own documents rather than whatever the question's words hit.
        q = ("refusal refused out of distribution unknown risk" if row.get("status") != "OK"
             else "review tier AUTO_OK REVIEW REFUSE three tiers")
        for d in _doc_sources(run, q, 1):
            yield _ev("source", source=_public(d))
            lines.append(f"\nWhat the tier means and how it is set is documented in the project [{d.source_id}].")
    lines.append("\nAll values above are read from the archive, not generated; no language model wrote them.")
    yield _ev("status", stage="checking", label="Checking evidence", request_id=run.request_id)
    yield _ev("done", answer=_answer(run, route, "\n".join(lines), "completed", claims=claims).model_dump(mode="json"))


# ---------------------------------------------------------------- web
def _search(run: Run, route: RouteDecision) -> Iterator[dict]:
    s = run.settings
    q = run.req.message if not route.required_geo_scope or route.required_geo_scope in run.req.message.lower() \
        else f"{run.req.message} {route.required_geo_scope}"
    plans: list[tuple[str, dict]] = []
    if route.intent == Intent.SAFETY_CRITICAL:
        plans.append((q, {"topic": "news", "include_domains": ev.SAFETY_DOMAINS}))
        plans.append((q + " India Meteorological Department", {"topic": "news"}))
    elif route.time_scope == "current":
        plans.append((q, {"topic": "news"}))
        plans.append((q + " official", {"topic": "general"}))
    else:
        plans.append((q, {"topic": "general"}))
    results: list[tuple[int, dict]] = []
    seen: set[str] = set()
    for query, opts in plans[: s.max_searches]:
        if run.expired():
            run.warnings.append("The time budget ran out before every search finished.")
            break
        yield _ev("status", stage="searching", label="Searching", request_id=run.request_id)
        ttl = 300 if route.time_scope == "current" else 6 * 3600
        key = hashlib.sha256(json.dumps([query.lower().strip(), route.required_geo_scope, route.time_scope,
                                         s.search_provider, opts, CONFIG_VERSION]).encode()).hexdigest()
        hits = SEARCH_CACHE.get(key)
        if hits is None:
            hits = run.deps.search.search(query, max_results=s.max_results, deadline=run.deadline, **opts)
            SEARCH_CACHE.put(key, hits, ttl)
            REGISTRY.counter("fbd_research_search_calls_total", "Web searches sent")
        else:
            REGISTRY.counter("fbd_research_search_cache_hits_total", "Web searches answered from cache")
        run.searched.append(query)
        for rank, h in enumerate(hits):
            try:
                urlsafe.validate(h["url"], resolver=lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 0))])
            except urlsafe.UnsafeURL:
                continue   # literal private IPs, bad schemes, credentials: dropped before anything else
            c = urlsafe.canonical(h["url"])
            if c in seen:
                continue
            seen.add(c)
            results.append((rank, {**h, "canonical": c}))
        # Enough authoritative material already: do not spend the second search.
        if sum(1 for _, h in results if ev.tier_for(_host(h["url"])) in (TrustTier.OFFICIAL,
                                                                       TrustTier.PRIMARY_RESEARCH)) >= 2:
            break

    yield _ev("status", stage="reading", label="Reading sources", request_id=run.request_id)
    docs: list[tuple[int, SourceDocument]] = []
    for rank, h in results:
        host = _host(h["url"])
        docs.append((rank, SourceDocument(
            source_id="", canonical_url=h["canonical"], title=h.get("title") or host, publisher=host,
            published_at=h.get("published_at"), retrieved_at=run.deps.clock(),
            content_hash=ev.content_hash(h.get("snippet", "")), extracted_text=h.get("snippet", ""),
            excerpt=(h.get("snippet") or "")[:280], trust_tier=ev.tier_for(host), fetch_status="snippet_only")))
    # Read the best few pages in full; the rest stay as search snippets.
    docs.sort(key=lambda p: -ev.score(p[1], p[0], route.intent))
    for _rank, d in docs[: s.max_fetches]:
        if run.expired() or not run.deps.fetch:
            break
        try:
            page = run.deps.fetch(d.canonical_url)
            if page.get("text"):
                d.extracted_text = page["text"]
                d.excerpt = page["text"][:280]
                d.title = page.get("title") or d.title
                d.published_at = d.published_at or page.get("published_at")
                d.publisher = page.get("site_name") or d.publisher
                d.content_hash = ev.content_hash(page["text"])
                d.fetch_status = "fetched"
        except urlsafe.UnsafeURL as exc:
            d.fetch_status = "blocked"
            d.flags.append(f"not fetched: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad page does not sink the answer
            d.fetch_status = "failed"
            d.flags.append(f"fetch failed ({type(exc).__name__}); search snippet used")
    for _rank, d in docs:
        ev.screen(d)
    docs.sort(key=lambda p: -ev.score(p[1], p[0], route.intent))
    for _rank, d in docs[: s.max_documents]:
        yield _ev("source", source=_public(run.add(d)))


def _synthesise(run: Run, route: RouteDecision) -> Iterator[dict]:
    usable = [d for d in run.sources if not any(f.startswith("injection") for f in d.flags)]
    if not usable:
        msg = ("I could not find usable sources for this, so I will not answer from memory."
               + (" Check the India Meteorological Department (mausam.imd.gov.in) and your State Disaster "
                  "Management Authority directly." if route.safety_level == "critical" else ""))
        yield _ev("done", answer=_answer(run, route, msg, "unable_to_verify").model_dump(mode="json"))
        return
    llm = run.deps.llm
    if llm is None:
        yield from _no_model(run, route, "No language model is configured on the server.")
        return
    yield _ev("status", stage="writing", label="Writing answer", request_id=run.request_id)
    today = run.deps.clock()[:10]
    user = prompt.user_message(run.req.message, route.intent.value, route.required_geo_scope, route.time_scope,
                               ev.packet(usable), today)
    result, fallback = None, False
    try:
        result = llm.generate(prompt.SYSTEM, user, max_tokens=run.settings.max_output_tokens, deadline=run.deadline)
    except ProviderFailure as exc:
        REGISTRY.counter("fbd_research_provider_errors_total", "LLM provider failures",
                         labels={"provider": exc.provider, "code": exc.code})
        if run.deps.fallback_llm is not None and not run.expired():
            yield _ev("warning", code="fallback", message=(
                f"The primary model failed ({exc.code}); the configured fallback "
                f"({run.deps.fallback_name or 'secondary'}) was used."))
            try:
                result = run.deps.fallback_llm.generate(prompt.SYSTEM, user,
                                                        max_tokens=run.settings.max_output_tokens,
                                                        deadline=run.deadline)
                fallback = True
                REGISTRY.counter("fbd_research_fallback_total", "Answers written by the fallback model")
            except ProviderFailure as exc2:
                yield from _no_model(run, route, f"Both configured models failed ({exc.code}, {exc2.code}).")
                return
        else:
            yield from _no_model(run, route, f"The model failed ({exc.code}: {exc}).",
                                 retryable=exc.retryable)
            return
    yield _ev("status", stage="checking", label="Checking evidence", request_id=run.request_id)
    ids = {d.source_id for d in usable}
    checked = validate.check(result.text, ids, ev.numbers_in(usable), [today, *ev.header_dates(usable)])
    if checked.removed_citations:
        REGISTRY.counter("fbd_research_citation_rejections_total", "Citations removed as not in the packet",
                         by=len(checked.removed_citations))
        run.warnings.append(f"{len(checked.removed_citations)} citation(s) to sources that were not retrieved "
                            "were removed.")
    if not checked.ok:
        REGISTRY.counter("fbd_research_verification_failures_total", "Answers withheld by validation")
        why = "; ".join(v.split(":")[0] for v in checked.violations)[:200]
        yield _ev("warning", code="verification_failed", message=f"The written answer failed verification ({why}).")
        yield _ev("done", answer=_answer(
            run, route, "The answer the model wrote did not pass verification, so it is not shown. "
            "The sources it was given are listed below; read them directly.", "unable_to_verify",
            llm=result, fallback=fallback, flags=[v[:160] for v in checked.violations]).model_dump(mode="json"))
        return
    out = checked.output
    if out.needs_clarification:
        MEMORY.put(run.req.conversation_id, {"question": run.req.message, "intent": route.intent.value})
        yield _ev("done", answer=_answer(run, route, out.clarification_question or out.answer, "clarification",
                                         llm=result, fallback=fallback, clarify=True).model_dump(mode="json"))
        return
    web = route.intent in (Intent.CURRENT_WEB_RESEARCH, Intent.SAFETY_CRITICAL)
    status = "completed"
    if not checked.cited_ids and web:
        status = "unable_to_verify"
    elif checked.removed_citations or any(c.support_level != "cited" for c in checked.claims) or out.limitations:
        status = "partial"
    yield _ev("done", answer=_answer(run, route, out.answer, status, claims=checked.claims,
                                     key_points=out.key_points, limitations=out.limitations, llm=result,
                                     fallback=fallback).model_dump(mode="json"))


def _no_model(run: Run, route: RouteDecision, why: str, retryable: bool = False) -> Iterator[dict]:
    """Sources without prose: what was found, honestly, when nothing can write it up."""
    run.warnings.append(why)
    lines = ["No summary could be written, so here is what was found:"]
    for d in run.sources[:5]:
        lines.append(f"- {d.title} [{d.source_id}]")
    if route.safety_level == "critical":
        lines.append("\nFor warnings, go to the India Meteorological Department and your State Disaster "
                     "Management Authority directly.")
    status = "partial" if run.sources else "unable_to_verify"
    yield _ev("done", answer=_answer(run, route, "\n".join(lines), status).model_dump(mode="json"),
              retryable=retryable)


def _docs_only(run: Run, route: RouteDecision) -> Iterator[dict]:
    """Method and how-to questions from the project's documents. With a model,
    a validated synthesis; without one, the passages themselves."""
    yield _ev("status", stage="reading", label="Reading project documents", request_id=run.request_id)
    for d in _doc_sources(run, run.req.message, 3):
        yield _ev("source", source=_public(d))
    if route.needs_web_search and run.deps.search is not None:
        yield from _search(run, route)
    if not run.sources:
        yield _ev("done", answer=_answer(run, route, "Nothing in the project's documents matches this question. "
                                         "Try naming the view, control or term you mean.", "unable_to_verify"
                                         ).model_dump(mode="json"))
        return
    if run.deps.llm is None:
        lines = ["From the project's own documents:"]
        for d in run.sources[:3]:
            lines.append(f"\n**{d.title}** [{d.source_id}]\n\n> " + d.excerpt)
        yield _ev("done", answer=_answer(run, route, "\n".join(lines), "partial",
                                         limitations=["No language model is configured, so these are excerpts, "
                                                      "not a written answer."]).model_dump(mode="json"))
        return
    yield from _synthesise(run, route)


# ------------------------------------------------------------------ entry
def run(req: ChatRequest, settings: ResearchSettings, deps: Deps | None = None) -> Iterator[dict]:
    deps = deps or default_deps(settings)
    r = Run(req=req, settings=settings, deps=deps)
    digest = hashlib.sha256(req.message.encode()).hexdigest()[:10]

    # A short reply to "which place?" completes the question it answers.
    pending = MEMORY.get(req.conversation_id)
    if pending and len(req.message.split()) <= 6 and not req.context_refs:
        req = req.model_copy(update={"message": f"{pending['question']} ({req.message})"})
        r.req = req
        MEMORY.forget(req.conversation_id)

    route = router.route(req.message, req.context_refs, req.web_mode)
    REGISTRY.counter("fbd_research_requests_total", "Research questions", labels={"intent": route.intent.value})
    yield _ev("status", stage="routing", label="Understanding the question", request_id=r.request_id,
              route=route.model_dump(mode="json"))
    final = None
    try:
        for e in _dispatch(r, route):
            if e["event"] == "done":
                final = e["data"]["answer"]
            yield e
    except ProviderFailure as exc:
        REGISTRY.counter("fbd_research_errors_total", "Research failures", labels={"code": exc.code})
        yield _ev("error", code=exc.code, message=f"{exc.provider}: {str(exc).split(': ', 1)[-1]}",
                  retryable=exc.retryable, request_id=r.request_id)
    except Exception as exc:  # noqa: BLE001 - never a stack trace to the client
        REGISTRY.counter("fbd_research_errors_total", "Research failures", labels={"code": "internal"})
        LOG.exception("research_failure request_id=%s", r.request_id)
        yield _ev("error", code="internal", message="Something went wrong on the server; nothing was answered.",
                  retryable=True, request_id=r.request_id)
    finally:
        log_event(LOG, "research_answer", request_id=r.request_id, intent=route.intent.value,
                  message_sha=digest, message_len=len(req.message), status=(final or {}).get("retrieval_status"),
                  sources=len(r.sources), searches=len(r.searched),
                  latency_ms=int((time.monotonic() - r.t0) * 1000))


def _dispatch(r: Run, route: RouteDecision) -> Iterator[dict]:
    req, s = r.req, r.settings
    if route.clarification:
        MEMORY.put(req.conversation_id, {"question": req.message, "intent": route.intent.value})
        yield _ev("done", answer=_answer(r, route, route.clarification, "clarification",
                                         clarify=True).model_dump(mode="json"))
        return
    if req.context_refs and "get_bulletin" in route.allowed_tools:
        yield from _explain_row(r, route)
        return
    if route.intent in (Intent.PROJECT_EXPLANATION, Intent.HOW_TO_USE_PRODUCT, Intent.SCIENTIFIC_BACKGROUND,
                        Intent.DATA_COMPARISON):
        yield from _docs_only(r, route)
        return
    # CURRENT_WEB_RESEARCH and SAFETY_CRITICAL need the web.
    if not route.needs_web_search or not s.web or r.deps.search is None:
        why = ("Web search is off in this build" if not s.web or req.web_mode == "off"
               else "No search provider is configured")
        msg = (f"{why}, so I cannot check current information and I will not answer it from memory."
               + (" For advisories, go to the India Meteorological Department (mausam.imd.gov.in) and your "
                  "State Disaster Management Authority." if route.safety_level == "critical" else ""))
        yield _ev("done", answer=_answer(r, route, msg, "unable_to_verify").model_dump(mode="json"))
        return
    ok, why = r.deps.search.available() if hasattr(r.deps.search, "available") else (True, "")
    if not ok:
        yield _ev("done", answer=_answer(r, route, f"Web search is unavailable: {why}. I will not answer current "
                                         "questions from memory.", "unable_to_verify").model_dump(mode="json"))
        return
    yield from _search(r, route)
    yield from _synthesise(r, route)
