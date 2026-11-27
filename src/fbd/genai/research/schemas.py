"""Typed contract for the research assistant: request, route, evidence, answer.

Contract first (Loop C): the pipeline, the API and the page all speak these
shapes, and every one is validated at the boundary it crosses. Source ids are
issued by the server, per request, and an answer may cite only those; a URL on
screen can only be one the server retrieved.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
REGION_RE = r"^[A-Z][A-Z0-9_]{1,63}$"


class Intent(str, Enum):
    PROJECT_EXPLANATION = "PROJECT_EXPLANATION"
    CURRENT_WEB_RESEARCH = "CURRENT_WEB_RESEARCH"
    SCIENTIFIC_BACKGROUND = "SCIENTIFIC_BACKGROUND"
    DATA_COMPARISON = "DATA_COMPARISON"
    HOW_TO_USE_PRODUCT = "HOW_TO_USE_PRODUCT"
    SAFETY_CRITICAL = "SAFETY_CRITICAL"
    OUT_OF_SCOPE_OR_AMBIGUOUS = "OUT_OF_SCOPE_OR_AMBIGUOUS"


class ContextRef(BaseModel):
    """A pointer to a project record, never free text from the page.

    "Explain this reading" sends these three validated fields; the server looks
    the row up itself. The DOM's own text never reaches the model.
    """

    region_id: str = Field(..., pattern=REGION_RE)
    init_date: str = Field(..., pattern=DATE_RE)
    lead_day: int = Field(..., ge=1, le=10)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{8,64}$")
    context_refs: list[ContextRef] = Field(default_factory=list, max_length=3)
    web_mode: Literal["auto", "on", "off"] = "auto"
    response_mode: Literal["stream", "complete"] = "stream"

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message is empty")
        return v


class RouteDecision(BaseModel):
    intent: Intent
    needs_web_search: bool
    needs_internal_data: bool
    safety_level: Literal["none", "elevated", "critical"] = "none"
    required_geo_scope: str | None = None
    time_scope: Literal["current", "historical", "timeless"] = "timeless"
    allowed_tools: list[str] = Field(default_factory=list)
    #: Short and user-safe: which rule fired. Not a reasoning trace.
    rationale: str
    clarification: str | None = None


class TrustTier(str, Enum):
    OFFICIAL = "official"            # government / meteorological agency, by allowlist
    PRIMARY_RESEARCH = "primary_research"
    REFERENCE = "reference"          # recognised data providers, documentation
    PROJECT = "project"              # this system's own records and docs
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    canonical_url: str
    domain: str
    snippet: str = ""
    published_at: str | None = None
    retrieved_at: str
    provider: str
    rank: int


class SourceDocument(BaseModel):
    source_id: str                  # "S1", "S2", ... issued by the server
    canonical_url: str
    title: str
    publisher: str
    published_at: str | None = None # never invented: None when the page gives none
    retrieved_at: str
    content_hash: str
    extracted_text: str = Field("", exclude=True)   # sent to the model, not to the browser
    excerpt: str = ""
    trust_tier: TrustTier
    fetch_status: Literal["fetched", "snippet_only", "internal", "blocked", "failed"]
    flags: list[str] = Field(default_factory=list)


class ClaimEvidence(BaseModel):
    claim_id: str
    claim_text: str
    source_ids: list[str]
    support_level: Literal["cited", "unsupported"] = "cited"
    notes: str | None = None


RetrievalStatus = Literal["completed", "partial", "unable_to_verify", "clarification", "refused"]


class AssistantAnswer(BaseModel):
    answer_markdown: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[SourceDocument] = Field(default_factory=list)
    claims: list[ClaimEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    retrieval_status: RetrievalStatus
    route: RouteDecision
    searched: list[str] = Field(default_factory=list, description="the queries actually sent")
    model_id: str | None = None
    provider_id: str | None = None
    fallback_used: bool = False
    created_at: str
    latency_ms: int
    usage: dict = Field(default_factory=dict)
    safety_flags: list[str] = Field(default_factory=list)
    safety_note: str | None = None
    needs_clarification: bool = False
    authoritative: bool = Field(
        False, description="Always false: this assistant summarises sources and the archive.")


class ModelOutput(BaseModel):
    """What the LLM must return. Parsed strictly; one repair attempt at most."""

    answer: str = Field(..., min_length=1, max_length=6000)
    key_points: list[str] = Field(default_factory=list, max_length=6)
    claims: list[dict] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=6)
    needs_clarification: bool = False
    clarification_question: str | None = None
