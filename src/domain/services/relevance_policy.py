"""
RelevancePolicy — STEP 6.1.

When no valid candidate exists for a source element, we must NOT default to
EVOLVE automatically (explicit rule in the mission). This module implements a
first, honestly-scoped strategy: configurable technical-field patterns +
minimum semantic evidence + optional LLM opinion. It is NOT a trained
classifier -- that is out of scope for this phase and is documented as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.domain.entities.experimental import RelevanceResult

# Default patterns for fields that are almost always technical/infrastructure
# rather than a meaningful new business concept. Configurable, not hardcoded
# logic buried deep in a decision method.
DEFAULT_TECHNICAL_PATTERNS: List[str] = [
    r"^_", r"^__", r"(^|_)meta$", r"(^|_)metadata$",
    r"(^|_)raw$", r"(^|_)tmp$", r"(^|_)temp$", r"(^|_)debug$",
    r"(^|_)internal$", r"(^|_)cache$", r"(^|_)checksum$", r"(^|_)hash$",
    r"(^|_)etag$", r"(^|_)version$", r"(^|_)schema_version$",
    r"(^|_)__v$", r"(^|_)_id$" ,  # mongo-style internal ids handled separately by key logic
]


@dataclass
class RelevancePolicy:
    """
    Decide whether a source element with no valid candidate is a genuinely
    new/meaningful concept (EVOLVE) or an irrelevant/technical field (REJECT).
    """
    technical_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_TECHNICAL_PATTERNS))
    min_semantic_evidence: float = 0.3
    """Minimum embedding/semantic score below which we consider there is not
    enough evidence the field is even conceptually meaningful."""
    ambiguous_band: float = 0.15
    """If the semantic score sits within this margin of min_semantic_evidence
    on either side, we return an ambiguous (REVIEW-eligible) result instead
    of a hard True/False."""
    llm_relevance_fn: Optional[Callable[[str], Optional[bool]]] = None
    """Optional callable(source_name) -> True/False/None for LLM-based
    relevance judgment. None means "LLM abstained" or is unavailable. This is
    injected rather than hardwired so tests don't require a live LLM."""

    def __post_init__(self):
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.technical_patterns]

    def assess(self, source_name: str, semantic_evidence: float) -> RelevanceResult:
        """
        Args:
            source_name: dotted path / attribute name of the source element.
            semantic_evidence: best available signal of semantic meaningfulness
                for this field in isolation (e.g. best embedding similarity to
                *any* domain vocabulary, not just schema candidates -- caller's
                responsibility to supply something reasonable; 0.0 if unknown).
        """
        leaf = source_name.split(".")[-1] if "." in source_name else source_name

        for pattern in self._compiled:
            if pattern.search(leaf):
                return RelevanceResult(
                    relevant=False,
                    confidence=0.9,
                    rationale=f"Matches technical-field pattern '{pattern.pattern}'",
                )

        llm_opinion = self.llm_relevance_fn(source_name) if self.llm_relevance_fn else None
        if llm_opinion is not None:
            return RelevanceResult(
                relevant=llm_opinion,
                confidence=0.7,
                rationale="LLM relevance judgment"
                + (" (relevant)" if llm_opinion else " (not relevant)"),
            )

        lower_bound = self.min_semantic_evidence - self.ambiguous_band
        upper_bound = self.min_semantic_evidence + self.ambiguous_band

        if lower_bound <= semantic_evidence <= upper_bound:
            return RelevanceResult(
                relevant=False,
                confidence=0.5,
                rationale=(
                    f"Semantic evidence ({semantic_evidence:.2f}) is within the ambiguous band "
                    f"[{lower_bound:.2f}, {upper_bound:.2f}] around the minimum threshold "
                    f"({self.min_semantic_evidence:.2f}); flagged for human REVIEW rather than "
                    "an automatic EVOLVE/REJECT split."
                ),
            )

        if semantic_evidence >= upper_bound:
            return RelevanceResult(
                relevant=True,
                confidence=min(1.0, semantic_evidence),
                rationale=f"Semantic evidence ({semantic_evidence:.2f}) exceeds threshold "
                          f"({self.min_semantic_evidence:.2f}); treated as a meaningful new concept.",
            )

        return RelevanceResult(
            relevant=False,
            confidence=1.0 - semantic_evidence,
            rationale=f"Semantic evidence ({semantic_evidence:.2f}) below threshold "
                      f"({self.min_semantic_evidence:.2f}); treated as irrelevant/technical.",
        )

    def is_ambiguous(self, result: RelevanceResult) -> bool:
        """Helper for DecisionPolicy: an ambiguous relevance call (confidence
        0.5, from the middle band) should route to REVIEW rather than being
        forced into EVOLVE or REJECT."""
        return abs(result.confidence - 0.5) < 1e-9