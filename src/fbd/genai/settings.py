"""Feature flags and model configuration for the GenAI layer.

Everything here is **default OFF**.  D-015 makes that binding: the air-gapped
serving path is the default and must keep passing its tests with no network,
so every switch below starts disabled and has to be turned on deliberately.

Nothing in this module imports boto3 or the anthropic SDK.  Importing
``fbd.genai.settings`` must stay free of network side effects and must not
fail on a machine with no AWS credentials -- which is exactly the machine this
was written on (see the verification boundary in D-015).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag.  Absent means OFF."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Model identifiers
# --------------------------------------------------------------------------
# First-party Claude API id.  Bedrock takes the same id with an "anthropic."
# prefix, which BedrockSettings.model_id applies -- do not hardcode the
# prefixed form in two places.
CLAUDE_MODEL = "claude-opus-5"

# Cheap model for the narration path, where the job is rewriting a fixed set of
# TreeSHAP reasons into a sentence and frontier reasoning is not needed.
CLAUDE_MODEL_CHEAP = "claude-haiku-4-5"

DEFAULT_REGION = "us-east-1"

# Local model served by Ollama.  3B is deliberate: the assistant's job is
# narration over numbers the pipeline already computed and routing read-only
# tool calls, not open-ended reasoning.  It fits in ~2 GB of VRAM and keeps the
# whole loop inside the forecaster's window.
LOCAL_MODEL = "llama3.2:3b"

# Which backend serves the assistant.  "local" is the default because it needs
# no account, no key and no spend, and because it preserves the air-gap
# property the rest of the system is built around (D-019).
DEFAULT_PROVIDER = "local"


@dataclass(frozen=True)
class LocalSettings:
    """How to reach a local Ollama model.  No credentials, no cost."""

    host: str = field(
        default_factory=lambda: os.environ.get("FBD_OLLAMA_HOST", "http://localhost:11434")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("FBD_LOCAL_MODEL", LOCAL_MODEL)
    )
    max_tokens: int = 1024
    timeout: float = 120.0

    @property
    def model_id(self) -> str:
        """Local tags carry no vendor prefix."""
        return self.model


@dataclass(frozen=True)
class BedrockSettings:
    """How to reach Claude on Amazon Bedrock.

    Uses the Mantle client (the Messages-API Bedrock endpoint), not the legacy
    bedrock-runtime InvokeModel path.
    """

    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", DEFAULT_REGION))
    model: str = CLAUDE_MODEL
    max_tokens: int = 4096
    # "low" is deliberate: the LLM's job here is narration and routing over
    # numbers the model already computed, not open-ended reasoning.  Effort
    # buys nothing on that task and costs latency inside a 45-minute
    # forecaster window.
    effort: str = "low"

    @property
    def model_id(self) -> str:
        """Bedrock model ids carry an ``anthropic.`` prefix."""
        return f"anthropic.{self.model}"


@dataclass(frozen=True)
class GenAISettings:
    """Master switchboard.  Every capability is opt-in."""

    # Master kill switch.  With this OFF nothing in fbd.genai will attempt a
    # network call, and the API exposes no GenAI routes.
    enabled: bool = field(default_factory=lambda: _flag("FBD_GENAI_ENABLED"))
    # Narrate existing TreeSHAP reasons into prose.  Never generates numbers.
    narration: bool = field(default_factory=lambda: _flag("FBD_GENAI_NARRATION"))
    # Let the model call read-only tools over the bulletin store.
    tool_calling: bool = field(default_factory=lambda: _flag("FBD_GENAI_TOOLS"))
    # Retrieval over the project's own documentation.  Local index, no network.
    rag: bool = field(default_factory=lambda: _flag("FBD_GENAI_RAG"))
    # Refuse to run at all unless guardrails are active.  Defaults to True and
    # should never be set False outside a guardrail unit test.
    require_guardrails: bool = True

    # Which backend serves the assistant: "local" (Ollama, default) or
    # "bedrock" (managed cloud, needs credentials and funding).
    provider: str = field(
        default_factory=lambda: os.environ.get("FBD_GENAI_PROVIDER", DEFAULT_PROVIDER)
    )

    bedrock: BedrockSettings = field(default_factory=BedrockSettings)
    local: LocalSettings = field(default_factory=LocalSettings)

    @property
    def provider_supported(self) -> bool:
        """False if ``provider`` names a backend this build cannot construct.

        Kept separate from ``backend`` because the health path must never
        raise: an operator who mistypes FBD_GENAI_PROVIDER should see the
        mistake reported on /api/health, not a stack trace, and should not be
        shown a fallback backend as though it were the one they asked for.
        """
        from fbd.genai.providers import is_supported

        return is_supported(self.provider)

    @property
    def backend(self):
        """Settings for whichever provider is selected.

        An unrecognised name falls back to the local shape so callers have
        something to read limits off; ``providers.build`` is what rejects it,
        loudly, before any request is made.
        """
        from fbd.genai.providers import normalise

        return self.bedrock if normalise(self.provider) == "bedrock" else self.local

    def active(self) -> bool:
        """True only if the master switch and at least one capability are on."""
        return self.enabled and (self.narration or self.tool_calling or self.rag)

    def describe(self) -> dict:
        """Serialisable summary for /api/health, so the operator can see what
        is live without reading environment variables off the host."""
        return {
            "enabled": self.enabled,
            "narration": self.narration,
            "tool_calling": self.tool_calling,
            "rag": self.rag,
            "guardrails_required": self.require_guardrails,
            "provider": self.provider if self.enabled else None,
            # Reported verbatim so a typo in FBD_GENAI_PROVIDER is visible here
            # rather than silently presenting the fallback as the real backend.
            "provider_supported": self.provider_supported if self.enabled else None,
            "model": self.backend.model if self.enabled else None,
            # Only a managed cloud backend has a region; a local model does not.
            "region": self.bedrock.region if (self.enabled and self.backend is self.bedrock) else None,
            "offline_capable": self.provider_supported and self.backend is self.local,
            "offline_default": not self.enabled,
        }


def load() -> GenAISettings:
    """Build settings from the environment.  Cheap; call it per request."""
    return GenAISettings()
