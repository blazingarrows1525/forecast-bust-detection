"""Pluggable LLM backends for the assistant layer.

Why this exists
---------------
The assistant was originally wired directly to one managed cloud backend
(D-015). That made the whole GenAI layer un-runnable without a funded cloud
account, which is a poor property for a system whose headline claim is that it
runs air-gapped with the network unplugged.

Every provider here returns the **same response shape** -- the Messages-API
contract that ``fbd.genai.agent.run`` already consumes:

    response.stop_reason            "end_turn" | "tool_use" | "refusal"
    response.content                list of blocks
    block.type                      "text" | "tool_use"
    block.text                      (text blocks)
    block.id / .name / .input       (tool_use blocks)

Because the contract is fixed, the guardrails, tool dispatch, injection
screening and numeric-grounding checks are provider-agnostic: swapping the
backend cannot weaken them. The existing test suite exercises the agent against
an in-process fake that satisfies the same contract, so it covers every
provider by construction.

Providers
---------
``local``    Ollama on localhost. No account, no key, no spend, works offline.
             The default, because it preserves the air-gap property.
``bedrock``  Managed cloud backend (D-015). Requires credentials and funding.

There is deliberately no third "disabled" provider.  ``FBD_GENAI_ENABLED=0`` is
already the off switch, it is the default, and it is the one CI asserts on
(D-015).  A second way to say "off" would be a second thing to get wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    """A plain-text content block, shaped like the Messages API."""

    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    """A tool-call content block, shaped like the Messages API."""

    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Response:
    """Provider-agnostic response object consumed by ``agent.run``."""

    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    stop_details: Any = None
    # Non-contract diagnostics, useful for the health endpoint and for
    # reporting real latency/cost in the writeup.
    usage: dict = field(default_factory=dict)
    provider: str = ""
    model: str = ""


class ProviderError(RuntimeError):
    """Raised when a backend is reachable but returned something unusable."""


#: Provider names ``build`` will accept.  ``ollama`` is an alias for ``local``.
SUPPORTED = ("local", "bedrock")
_ALIASES = {"ollama": "local"}


def normalise(provider: str) -> str:
    """Canonical provider name, or the raw value if it is not one we serve."""
    name = (provider or "").strip().lower()
    return _ALIASES.get(name, name)


def is_supported(provider: str) -> bool:
    """True if ``build`` would accept this provider name."""
    return normalise(provider) in SUPPORTED


def build(provider: str, settings) -> Any:
    """Return a client exposing ``.messages.create(...)`` for the named provider."""
    name = normalise(provider)
    if name == "local":
        from fbd.genai.providers.ollama import OllamaClient

        return OllamaClient(settings)
    if name == "bedrock":
        from fbd.genai.client import build_bedrock_client

        return build_bedrock_client(settings)
    raise ProviderError(
        f"unknown GenAI provider {provider!r}; expected one of: "
        f"{', '.join(SUPPORTED)}. To turn the layer off, unset "
        f"FBD_GENAI_ENABLED (the default) rather than naming a provider."
    )
