"""Guardrails for the GenAI layer.

D-015 makes two things binding, and this module is where they are enforced in
code rather than asserted in a prompt:

1. **The LLM never produces a bust probability.**  Every number in generated
   text must already appear in the grounded facts handed to the model.  A model
   that invents "roughly 80% chance" next to a disaster-management product is a
   liability, and a system prompt saying "do not invent numbers" is not a
   control -- it is a request.  This module checks the output and rejects it.

2. **The non-interference invariant (LOGIC.md 1.3) survives.**  The system is
   decision support.  Generated text may not issue directives to the public or
   to civil authorities; it advises a duty forecaster to look at something.

Guardrails run **client side, in-process**.  That is deliberate: a network
guardrail service that is unreachable fails open, and failing open is the one
behaviour this layer must never have.  Bedrock Guardrails can be layered on
top as defence in depth (see ``BEDROCK_GUARDRAIL_NOTE``), but the checks here
are the ones the invariant actually rests on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A Bedrock Guardrail (guardrailIdentifier / guardrailVersion on the request)
# is a *second* layer, applied server side by AWS.  It is complementary, not a
# replacement: it cannot know which numbers our model computed, so it cannot
# enforce invariant 1.  Configure it for the generic content-safety categories
# and keep these checks for the domain invariants.
BEDROCK_GUARDRAIL_NOTE = (
    "Bedrock Guardrails cover generic content safety. The numeric-grounding "
    "and non-interference invariants are domain-specific and are enforced here."
)

# Any run of digits, with optional decimal part and optional trailing percent.
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent)?", re.IGNORECASE)

# Numbers that are never risk claims and are safe to see in prose.  Lead days
# run 1-10, small counts appear in phrases like "three reasons".
_SAFE_SMALL_INTEGERS = set(range(0, 11))

# Phrasing that would turn decision support into an instruction.  The system
# advises a forecaster; it does not tell anyone to act.  LOGIC.md 1.3.
_AUTHORITY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bevacuat(e|ing|ion)\b",
        r"\bissue (a |an )?(red|orange|amber) (alert|warning)\b",
        r"\bwe (are )?(hereby )?(issuing|declaring)\b",
        r"\bthe public should\b",
        r"\bdo not travel\b",
        r"\bclose (the )?(schools|offices)\b",
        r"\bshut down\b",
        r"\bofficial (warning|bulletin) is\b",
    )
]

# Instructions arriving inside retrieved documents or tool output are data, not
# commands.  These are the shapes that most often smuggle one in.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |the )?(previous|prior|above) instructions",
        r"disregard (all |the )?(previous|prior|above)",
        r"you are now (a|an|in) ",
        r"system prompt\s*[:=]",
        r"<\s*/?\s*(system|instructions?)\s*>",
        r"\bnew instructions?\s*[:=]",
        r"reveal (your|the) (system )?prompt",
        r"act as (if )?(you are )?(an? )?(unrestricted|jailbroken|developer mode)",
    )
]


class GuardrailViolation(Exception):
    """Raised when generated text fails a hard invariant."""


@dataclass
class GuardrailReport:
    """Outcome of a guardrail pass.  Always inspect ``ok``."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)
    redacted_text: str | None = None

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise GuardrailViolation("; ".join(self.violations))


def _candidate_values(raw: str, is_percent: bool) -> list[float]:
    """Every reading of a numeric token that we are willing to call grounded.

    "70.6%" may legitimately restate a stored probability of 0.706, so both
    readings count as a match.
    """
    value = float(raw)
    readings = [value]
    if is_percent:
        readings.append(value / 100.0)
    else:
        readings.append(value * 100.0)
    return readings


def _is_grounded(raw: str, is_percent: bool, allowed: list[float], tol: float) -> bool:
    for reading in _candidate_values(raw, is_percent):
        for ok_value in allowed:
            if abs(reading - ok_value) <= tol:
                return True
            # Tolerate the model rounding a stored 0.7058 to 0.71 or 70.6%.
            if round(reading, 2) == round(ok_value, 2):
                return True
            if round(reading, 1) == round(ok_value, 1):
                return True
    return False


def check_numeric_grounding(
    text: str,
    allowed_numbers: list[float] | set[float],
    tol: float = 0.05,
) -> GuardrailReport:
    """Verify every number in ``text`` traces back to a grounded fact.

    ``allowed_numbers`` is whatever the caller actually computed and handed to
    the model -- probabilities, rainfall amounts, interval bounds, base rates.
    Anything else in the output is a number the model made up.
    """
    allowed = [float(v) for v in allowed_numbers]
    ungrounded: list[float] = []

    for match in _NUMBER_RE.finditer(text):
        raw, suffix = match.group(1), match.group(2)
        is_percent = suffix is not None
        value = float(raw)

        # Bare small integers (lead days, "three factors") are not risk claims.
        if not is_percent and value.is_integer() and int(value) in _SAFE_SMALL_INTEGERS:
            continue
        # Years are not risk claims either.
        if not is_percent and value.is_integer() and 1900 <= value <= 2100:
            continue

        if not _is_grounded(raw, is_percent, allowed, tol):
            ungrounded.append(value)

    if ungrounded:
        listed = ", ".join(str(v) for v in ungrounded)
        return GuardrailReport(
            ok=False,
            violations=[
                f"ungrounded numeric claim(s) in generated text: {listed}. "
                "Every number must come from the calibrated model, not the LLM."
            ],
            ungrounded_numbers=ungrounded,
        )
    return GuardrailReport(ok=True)


def check_authority(text: str) -> GuardrailReport:
    """Reject text that issues instructions instead of advising review."""
    hits = [p.pattern for p in _AUTHORITY_PATTERNS if p.search(text)]
    if hits:
        return GuardrailReport(
            ok=False,
            violations=[
                "generated text issues a directive, violating the "
                f"non-interference invariant (LOGIC.md 1.3): matched {hits}"
            ],
        )
    return GuardrailReport(ok=True)


def check_injection(text: str) -> GuardrailReport:
    """Flag instruction-shaped content arriving from retrieved data.

    Retrieved documents and tool results are data.  When one contains something
    aimed at the model, the correct behaviour is to surface it, not obey it.
    """
    hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
    if hits:
        return GuardrailReport(
            ok=False,
            violations=[f"possible prompt injection in untrusted content: {hits}"],
        )
    return GuardrailReport(ok=True)


def guard_output(text: str, allowed_numbers: list[float] | set[float]) -> GuardrailReport:
    """Full output pass: numeric grounding + non-interference.

    This is the function the serving path calls.  It combines every violation
    rather than short-circuiting, so an operator sees all of what went wrong.
    """
    violations: list[str] = []
    ungrounded: list[float] = []

    numeric = check_numeric_grounding(text, allowed_numbers)
    if not numeric.ok:
        violations.extend(numeric.violations)
        ungrounded.extend(numeric.ungrounded_numbers)

    authority = check_authority(text)
    if not authority.ok:
        violations.extend(authority.violations)

    return GuardrailReport(
        ok=not violations, violations=violations, ungrounded_numbers=ungrounded
    )


def sanitise_untrusted(text: str, max_chars: int = 4000) -> tuple[str, GuardrailReport]:
    """Prepare retrieved/tool content for inclusion in a prompt.

    Truncates, and reports (does not silently strip) anything instruction
    shaped, so the caller can decide whether to drop the passage entirely.
    """
    clipped = text[:max_chars]
    return clipped, check_injection(clipped)
