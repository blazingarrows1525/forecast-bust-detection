"""Configuration for the research assistant.  Everything defaults OFF.

The research assistant answers questions about a reading, the method, and --
only when an operator turns it on -- the public web. It sits behind the same
master switch as the rest of the GenAI layer (``FBD_GENAI_ENABLED``, D-015):
with that off there is no route that could attempt egress, which is what CI's
air-gap job asserts.  Web search needs a second, separate opt-in
(``FBD_RESEARCH_WEB``), because "the assistant may read the project's own
docs" and "the server may reach the internet" are different promises.

Model identifiers live in ONE place, ``PROVIDERS`` below, and each is
overridable by environment. They are the defaults this build was written
against; provider catalogues change, so ``docs/assistant-provider-decision.md``
records when each was last checked and the operator is expected to re-check
before a deployment. Nothing else in the code names a model.

Keys are read from the environment on the server and nowhere else. They never
reach a response body, a log line or the browser.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class ProviderSpec:
    """One LLM backend: how to reach it and what it costs to use."""

    name: str
    kind: str            # "gemini" | "openai_compat" | "ollama" | "mock"
    base_url: str
    key_env: str | None  # environment variable holding the key, or None
    model_env: str       # environment variable that overrides the model
    default_model: str
    #: "free" (a free tier exists for the default model), "paid" (metered from
    #: the first token), or "local" (no provider at all). A paid backend is
    #: never used as a fallback unless FBD_ASSISTANT_ALLOW_PAID=1.
    tier: str
    #: The provider's own statement of what it does with prompts on this tier,
    #: summarised in the decision record. Shown on /status so an operator does
    #: not have to go looking for it.
    data_note: str

    def model(self) -> str:
        return os.environ.get(self.model_env, self.default_model)

    def key(self) -> str | None:
        return os.environ.get(self.key_env) if self.key_env else None


# The registry. Defaults checked 2026-10-18; see docs/assistant-provider-decision.md.
PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        "gemini", "gemini", "https://generativelanguage.googleapis.com/v1beta",
        "GEMINI_API_KEY", "FBD_GEMINI_MODEL", "gemini-flash-latest", "free",
        "Free tier: Google states it may use prompts to improve its products. "
        "Paid tier: it does not. Do not send personal data on the free tier."),
    "openrouter": ProviderSpec(
        "openrouter", "openai_compat", "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY", "FBD_OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free", "free",
        "Routes to third-party hosts; data handling depends on the upstream "
        "provider chosen per request. Check the model page before enabling."),
    "groq": ProviderSpec(
        "groq", "openai_compat", "https://api.groq.com/openai/v1",
        "GROQ_API_KEY", "FBD_GROQ_MODEL", "llama-3.3-70b-versatile", "free",
        "Developer plan has per-model request and token limits; see the "
        "decision record for retention terms."),
    "local": ProviderSpec(
        "local", "ollama", "", None, "FBD_LOCAL_MODEL", "llama3.1:8b", "local",
        "Runs on this machine; nothing leaves it."),
    "mock": ProviderSpec(
        "mock", "mock", "", None, "FBD_MOCK_MODEL", "mock-1", "local",
        "Deterministic test double; no network."),
}

SEARCH_PROVIDERS = ("tavily", "mock")


@dataclass(frozen=True)
class ResearchSettings:
    """Switchboard and budgets. Read per request; cheap."""

    #: Master switch for this feature (on top of FBD_GENAI_ENABLED).
    enabled: bool = field(default_factory=lambda: _flag("FBD_GENAI_ENABLED") and _flag("FBD_RESEARCH"))
    #: Separate opt-in for any outbound web request.
    web: bool = field(default_factory=lambda: _flag("FBD_RESEARCH_WEB"))
    search_provider: str = field(default_factory=lambda: os.environ.get("FBD_SEARCH_PROVIDER", "tavily"))
    llm: str = field(default_factory=lambda: os.environ.get("FBD_ASSISTANT_LLM", "gemini"))
    #: An explicitly configured second backend. Empty means no fallback.
    fallback: str = field(default_factory=lambda: os.environ.get("FBD_ASSISTANT_FALLBACK", ""))
    allow_paid: bool = field(default_factory=lambda: _flag("FBD_ASSISTANT_ALLOW_PAID"))
    #: True when a trusted outbound proxy (Claude Code Network secrets, an API
    #: gateway) injects auth headers for the configured hosts, so the keys never
    #: reach this process. The code then neither checks for a local key nor
    #: sends an auth header; see providers._proxy_auth.
    proxy_auth: bool = field(default_factory=lambda: _flag("FBD_ASSISTANT_PROXY_AUTH"))

    # Budgets: every one is a hard cap, enforced in the pipeline.
    max_searches: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_SEARCHES", 2))
    max_results: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_RESULTS", 5))
    max_documents: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_DOCS", 6))
    max_fetches: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_FETCHES", 3))
    max_page_bytes: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_PAGE_BYTES", 1_500_000))
    max_page_chars: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_PAGE_CHARS", 5000))
    max_output_tokens: int = field(default_factory=lambda: _int("FBD_RESEARCH_MAX_OUTPUT_TOKENS", 1200))
    deadline_s: int = field(default_factory=lambda: _int("FBD_RESEARCH_DEADLINE_S", 45))
    per_client_per_minute: int = field(default_factory=lambda: _int("FBD_RESEARCH_RPM", 6))
    per_client_per_day: int = field(default_factory=lambda: _int("FBD_RESEARCH_RPD", 60))
    global_per_day: int = field(default_factory=lambda: _int("FBD_RESEARCH_GLOBAL_RPD", 400))

    def spec(self, name: str | None = None) -> ProviderSpec | None:
        return PROVIDERS.get((name or self.llm).strip().lower())

    def fallback_spec(self) -> ProviderSpec | None:
        """The configured fallback, only if policy allows it. Never implicit."""
        if not self.fallback:
            return None
        spec = self.spec(self.fallback)
        if spec is None or spec.name == (self.spec() or spec).name:
            return None
        if spec.tier == "paid" and not self.allow_paid:
            return None
        return spec

    def describe(self) -> dict:
        """What /status shows. Never includes a key; says only whether one is set."""
        primary, fb = self.spec(), self.fallback_spec()

        def row(s: ProviderSpec | None) -> dict | None:
            if s is None:
                return None
            has_local_key = (s.key() is not None) if s.key_env else None
            auth = "proxy" if (not s.key()) and self.proxy_auth and s.key_env else (
                "local-env" if has_local_key else "none")
            return {"provider": s.name, "model": s.model(), "tier": s.tier,
                    "key_configured": has_local_key, "auth": auth,
                    "data_note": s.data_note}

        search_key = bool(os.environ.get("TAVILY_API_KEY")) if self.web else None
        return {
            "enabled": self.enabled,
            "web_search": self.web,
            "search_provider": self.search_provider if self.web else None,
            "search_key_configured": search_key,
            "search_auth": ("proxy" if (self.web and not search_key and self.proxy_auth)
                            else ("local-env" if search_key else "none")) if self.web else None,
            "proxy_auth": self.proxy_auth,
            "llm": row(primary),
            "llm_known": primary is not None,
            "fallback": row(fb),
            "fallback_requested": self.fallback or None,
            "allow_paid": self.allow_paid,
            "budgets": {
                "max_searches": self.max_searches, "max_documents": self.max_documents,
                "max_fetches": self.max_fetches, "max_output_tokens": self.max_output_tokens,
                "deadline_s": self.deadline_s, "per_client_per_minute": self.per_client_per_minute,
                "per_client_per_day": self.per_client_per_day, "global_per_day": self.global_per_day,
            },
            "history": "kept in memory for 30 minutes to resolve follow-ups; never written to disk",
        }


def load() -> ResearchSettings:
    return ResearchSettings()
