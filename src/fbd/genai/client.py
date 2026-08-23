"""Bedrock client construction, with honest degradation.

**Verification boundary (D-015):** the machine this was written on has no AWS
CLI, no boto3 and no credentials.  Everything below is written against the
documented Bedrock Mantle client surface and unit-tested against a local fake;
none of it has been executed against real AWS.  ``availability()`` exists so
the system can *say* that at runtime instead of failing with an import error
three layers down.

Uses ``AnthropicBedrockMantle`` -- the Messages-API Bedrock endpoint -- rather
than the legacy bedrock-runtime InvokeModel path.
"""
from __future__ import annotations

from dataclasses import dataclass

from fbd.genai.settings import BedrockSettings


@dataclass
class Availability:
    """Why the GenAI layer can or cannot run right now."""

    ok: bool
    reason: str
    sdk_installed: bool = False

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "sdk_installed": self.sdk_installed}


def availability() -> Availability:
    """Check whether a Bedrock call could actually succeed.

    Deliberately does not make a network call -- this runs on the health path,
    and a health check that blocks on a remote endpoint is a liability.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return Availability(
            ok=False,
            reason=(
                "anthropic SDK not installed. `pip install anthropic[bedrock]` "
                "to enable the GenAI layer. The offline serving path is "
                "unaffected and remains the default."
            ),
        )

    try:
        from anthropic import AnthropicBedrockMantle  # noqa: F401
    except ImportError:
        return Availability(
            ok=False,
            sdk_installed=True,
            reason=(
                "anthropic is installed but AnthropicBedrockMantle is missing; "
                "the Bedrock extra is required: pip install 'anthropic[bedrock]'."
            ),
        )

    try:
        import boto3  # noqa: F401
    except ImportError:
        return Availability(
            ok=False,
            sdk_installed=True,
            reason="boto3 not installed; Bedrock auth needs it.",
        )

    session_ok, detail = _credentials_present()
    if not session_ok:
        return Availability(ok=False, sdk_installed=True, reason=detail)

    return Availability(ok=True, sdk_installed=True, reason="ready")


def _credentials_present() -> tuple[bool, str]:
    try:
        import boto3

        creds = boto3.Session().get_credentials()
    except Exception as exc:  # noqa: BLE001 - any boto3 failure means "no creds"
        return False, f"could not resolve AWS credentials: {exc}"
    if creds is None:
        return False, (
            "no AWS credentials found (no ~/.aws/credentials, no AWS_* environment "
            "variables, no instance role). Bedrock calls would fail."
        )
    return True, "credentials resolved"


def build_client(settings: BedrockSettings | None = None):
    """Construct the Bedrock Mantle client.

    Raises RuntimeError with an actionable message rather than an ImportError
    or a botocore stack trace, because this is reached from a request path.
    """
    settings = settings or BedrockSettings()
    check = availability()
    if not check.ok:
        raise RuntimeError(f"Bedrock unavailable: {check.reason}")

    from anthropic import AnthropicBedrockMantle

    return AnthropicBedrockMantle(aws_region=settings.region)
