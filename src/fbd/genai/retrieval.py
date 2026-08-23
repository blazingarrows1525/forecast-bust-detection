"""Local retrieval over the project's own documentation (the RAG tier).

Deliberately **not** an embedding service.  A hosted embedding endpoint would
put a network round trip inside the retrieval path and break the offline
default that D-015 preserves.  BM25 over the repository's markdown runs in
milliseconds with no dependency beyond the standard library, and for a corpus
of five engineering documents it is entirely adequate.

What it indexes is the point: LOGIC.md, DECISIONS.md, README.md and
PROJECT_BLUEPRINT.md are where every methodological choice and its evidence
lives.  That makes "why is Assam flagged when the ensemble is confident?" and
"why were the islands excluded?" answerable *with citations to our own decision
log* rather than from model memory.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fbd import config

# Documents worth retrieving over.  Order is irrelevant; provenance is not.
CORPUS_FILES = (
    "LOGIC.md",
    "DECISIONS.md",
    "README.md",
    "PROJECT_BLUEPRINT.md",
    "HANDOFF.md",
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "for", "it", "that", "this", "with", "as", "at", "by", "from",
    "we", "our", "not", "but", "if", "so", "than", "then", "which", "what",
}


def _tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Passage:
    """One retrievable chunk, carrying enough provenance to cite it."""

    doc: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc} :: {self.heading}" if self.heading else self.doc


def _split_sections(doc_name: str, raw: str) -> list[Passage]:
    """Chunk a markdown file on its headings.

    Heading-aligned chunks beat fixed-width windows here because these
    documents are already organised by decision (D-001, D-002, ...) and by
    numbered spec section, so a heading *is* the semantic boundary.
    """
    passages: list[Passage] = []
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            passages.append(Passage(doc=doc_name, heading=current_heading, text=body))

    in_fence = False
    for line in raw.splitlines():
        # Track fenced code blocks: a '#' inside one is a shell comment, not a
        # heading. Without this, every commented pipeline step in a bash block
        # becomes its own bogus section.
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if line.startswith("#") and not in_fence:
            flush()
            buffer = []
            current_heading = line.lstrip("#").strip()
        else:
            buffer.append(line)
    flush()
    return passages


class DocIndex:
    """A small BM25 index. Build once at startup, query per request."""

    K1 = 1.5
    B = 0.75

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self._tokens = [_tokenise(p.text + " " + p.heading) for p in passages]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._tf = [Counter(t) for t in self._tokens]

        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        n = max(len(passages), 1)
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query: str, k: int = 4) -> list[tuple[Passage, float]]:
        q_terms = _tokenise(query)
        if not q_terms or not self.passages:
            return []

        scored: list[tuple[Passage, float]] = []
        for i, passage in enumerate(self.passages):
            tf, length = self._tf[i], self._lengths[i]
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.K1 * (
                    1 - self.B + self.B * (length / (self._avg_len or 1.0))
                )
                score += idf * (freq * (self.K1 + 1)) / denom
            if score > 0:
                scored.append((passage, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def build_index(root: Path | None = None) -> DocIndex:
    """Index whichever corpus files are present.  Missing files are skipped."""
    base = root or config.ROOT
    passages: list[Passage] = []
    for name in CORPUS_FILES:
        path = base / name
        if not path.exists():
            continue
        passages.extend(_split_sections(name, path.read_text(encoding="utf-8")))
    return DocIndex(passages)


def format_context(hits: list[tuple[Passage, float]], max_chars: int = 1200) -> str:
    """Render hits as a citation-carrying context block.

    The passages are wrapped in an explicit data marker.  Retrieved text is
    reference material, never instructions -- guardrails.sanitise_untrusted is
    what actually screens it, but labelling the boundary in the prompt too
    costs nothing.
    """
    if not hits:
        return "(no matching project documentation)"
    blocks = []
    for passage, score in hits:
        body = passage.text[:max_chars].strip()
        blocks.append(f"[{passage.citation}] (relevance {score:.2f})\n{body}")
    return "\n\n---\n\n".join(blocks)
