"""Source policy, scoring, and the evidence packet the model is allowed to see.

Trust tiers are assigned by allowlist, never by what a page says about itself.
A page that claims to be official is still "unknown" unless its host is on the
list. Only OFFICIAL is labelled "official source" in the interface.

Scoring is deliberately simple and inspectable:

    score = authority + relevance + recency + extraction

where recency counts for current and safety questions and barely at all for
stable science (a 1998 paper on the Coriolis force is not stale). Secondary
sources may add context; they never outrank a primary source that disagrees.

The packet hands the model numbered sources between explicit
untrusted-content markers. Anything instruction-shaped in a source is flagged
by the existing injection screen and that source is withheld from the model,
with the reason shown to the user.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from fbd.genai.guardrails import check_injection
from fbd.genai.research.schemas import Intent, SourceDocument, TrustTier

# Suffix match on the host. Checked 2026-10-18; extend deliberately, in review.
OFFICIAL = (
    "imd.gov.in", "mausam.imd.gov.in", "ndma.gov.in", "nidm.gov.in", "cwc.gov.in", "incois.gov.in",
    "isro.gov.in", "mosdac.gov.in", "imdpune.gov.in", "tropmet.res.in", "ncmrwf.gov.in", "pib.gov.in",
    "gov.in", "nic.in", "wmo.int", "noaa.gov", "weather.gov", "nasa.gov", "ecmwf.int", "metoffice.gov.uk",
    "bom.gov.au", "jma.go.jp", "un.org", "undrr.org", "gdacs.org",
)
PRIMARY_RESEARCH = (
    "doi.org", "nature.com", "science.org", "agupubs.onlinelibrary.wiley.com", "rmets.onlinelibrary.wiley.com",
    "journals.ametsoc.org", "copernicus.org", "sciencedirect.com", "springer.com", "arxiv.org",
    "pnas.org", "iopscience.iop.org", "tandfonline.com",
)
REFERENCE = (
    "copernicus.eu", "weatherbench2.readthedocs.io", "earthdata.nasa.gov", "gpm.nasa.gov",
    "worldbank.org", "britannica.com", "ametsoc.org", "rmets.org",
)
# Official sources to try first for safety-critical questions (Tavily include_domains).
# Warning agencies only. pib.gov.in (the government press office) was here; with
# it, Tavily's news search answered "heavy rainfall warning for Assam" with
# unrelated press notes in live runs (2026-10-18). PIB stays OFFICIAL when found.
SAFETY_DOMAINS = ["mausam.imd.gov.in", "imd.gov.in", "ndma.gov.in", "cwc.gov.in", "incois.gov.in"]

AUTHORITY = {TrustTier.OFFICIAL: 4.0, TrustTier.PRIMARY_RESEARCH: 3.0, TrustTier.REFERENCE: 2.0,
             TrustTier.PROJECT: 4.0, TrustTier.SECONDARY: 1.0, TrustTier.UNKNOWN: 0.5}


def _matches(host: str, suffixes: tuple[str, ...]) -> bool:
    return any(host == s or host.endswith("." + s) for s in suffixes)


def tier_for(host: str) -> TrustTier:
    host = (host or "").lower()
    if _matches(host, OFFICIAL):
        return TrustTier.OFFICIAL
    if _matches(host, PRIMARY_RESEARCH):
        return TrustTier.PRIMARY_RESEARCH
    if _matches(host, REFERENCE):
        return TrustTier.REFERENCE
    if re.search(r"\.(edu|ac\.in|ac\.uk|res\.in)$", host):
        return TrustTier.PRIMARY_RESEARCH
    if host:
        return TrustTier.SECONDARY
    return TrustTier.UNKNOWN


def _age_days(published: str | None) -> float | None:
    if not published:
        return None
    try:
        d = datetime.fromisoformat(published.replace("Z", "+00:00")[:25])
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - d).total_seconds() / 86400)
    except ValueError:
        return None


def score(doc: SourceDocument, rank: int, intent: Intent) -> float:
    s = AUTHORITY[doc.trust_tier]
    s += max(0.0, 1.5 - 0.25 * rank)                     # the search provider's own relevance order
    age = _age_days(doc.published_at)
    if intent in (Intent.CURRENT_WEB_RESEARCH, Intent.SAFETY_CRITICAL):
        s += 2.0 if age is not None and age <= 3 else 1.0 if age is not None and age <= 14 else -0.5 if age is not None else 0.0
    s += 0.5 if doc.fetch_status == "fetched" else 0.0
    s -= 3.0 if "injection" in " ".join(doc.flags) else 0.0
    return round(s, 3)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def screen(doc: SourceDocument) -> SourceDocument:
    """Flag instruction-shaped content. A flagged source is shown to the user
    with the flag but its text is withheld from the model."""
    report = check_injection(doc.extracted_text + " " + doc.title)
    if not report.ok:
        doc.flags.append("injection: instruction-shaped text found in the page; not given to the model")
    return doc


def packet(docs: list[SourceDocument], max_chars_each: int = 3500) -> str:
    """The only source text the model sees, each part fenced as untrusted data."""
    blocks = []
    for d in docs:
        if any(f.startswith("injection") for f in d.flags):
            continue
        published = d.published_at or "not stated by the page"
        body = d.extracted_text[:max_chars_each].replace("<<<", "‹‹‹").replace(">>>", "›››")
        blocks.append(
            f"<<<SOURCE {d.source_id} | {d.trust_tier.value} | {d.publisher} | published: {published} "
            f"| retrieved: {d.retrieved_at}\nTITLE: {d.title}\n{body}\nEND SOURCE {d.source_id}>>>")
    return "\n\n".join(blocks) if blocks else "(no usable sources)"


def header_dates(docs: list[SourceDocument]) -> list[str]:
    """The published and retrieved times each packet header states: dates an answer may restate."""
    return [t for d in docs if not any(f.startswith("injection") for f in d.flags)
            for t in (d.published_at, d.retrieved_at) if t]


def numbers_in(docs: list[SourceDocument]) -> list[float]:
    """Every number any usable source contains: what an answer may restate."""
    out: list[float] = []
    for d in docs:
        if any(f.startswith("injection") for f in d.flags):
            continue
        for tok in re.findall(r"-?\d+(?:\.\d+)?", d.extracted_text + " " + d.title):
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out
