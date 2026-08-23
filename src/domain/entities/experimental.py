"""
Domain entities for the experimental pipeline (STEP 3-6.2).

See docs/experimental_architecture.md for the audit that justifies why each
of these is new vs. reused from the existing Stack A / Stack B code.

Design choices carried over from the audit:
- DecisionAction (ACCEPT/REVIEW/REJECT, in rag_schema.py) is left untouched.
  DecisionType (MATCH/EVOLVE/REVIEW/REJECT) is a NEW, separate enum. The two
  coexist temporarily: DecisionAction is Stack B's internal vocabulary,
  DecisionType is the paper's open-world vocabulary. `decision_action_to_type`
  below is the single explicit conversion point between them.
- CandidateMatch (rag_schema.py) already exists and is used inside Stack B's
  LLM/scoring flow. CandidateResult here is NOT a duplicate: it represents the
  raw *retrieval* output (source/target/rank/similarity/timing) before any
  scoring or LLM step, i.e. what AdvancedRAGRetriever produces. CandidateMatch
  represents a *scored* candidate further downstream. Both are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.domain.entities.rag_schema import DecisionAction


# ---------------------------------------------------------------------------
# STEP 3 — Candidate Retrieval
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateResult:
    """
    One retrieval candidate for a source element, BEFORE any semantic
    decision or MDE validation is applied. The retriever must never discard
    candidates after picking Top-1 — every candidate observed for a given
    source_element should be represented by one CandidateResult, with `rank`
    indicating its position in the retrieved list.
    """
    source_element: str
    target_element: str
    rank: int
    similarity: float
    retrieval_time_ms: float = 0.0
    is_gold: Optional[bool] = None  # populated only when ground truth is known (reproduction/eval)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# STEP 4 — Decision Type (open-world)
# ---------------------------------------------------------------------------

class DecisionType(str, Enum):
    """Open-world decision outcome for the experimental pipeline."""
    MATCH = "MATCH"
    EVOLVE = "EVOLVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


def decision_action_to_type(action: DecisionAction) -> DecisionType:
    """
    Explicit, single-point conversion from Stack B's closed-world
    DecisionAction to the open-world DecisionType.

    NOTE: DecisionAction has no EVOLVE equivalent (it predates open-world
    matching), so ACCEPT always maps to MATCH here. Producing an actual
    EVOLVE decision requires DecisionPolicy (STEP 6), not this mapping.
    """
    mapping = {
        DecisionAction.ACCEPT: DecisionType.MATCH,
        DecisionAction.REVIEW: DecisionType.REVIEW,
        DecisionAction.REJECT: DecisionType.REJECT,
    }
    return mapping[action]


# ---------------------------------------------------------------------------
# STEP 4.1 — Semantic Decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticDecision:
    """
    Result of semantic evaluation for ONE candidate, before MDE validation
    and before the final DecisionType is chosen.
    """
    candidate: CandidateResult
    embedding_score: float
    llm_score: float
    combined_score: float
    rationale: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# STEP 5 — MDE Validation
# ---------------------------------------------------------------------------

class MDEStatus(str, Enum):
    VALID = "VALID"
    VALID_WITH_TRANSFORMATION = "VALID_WITH_TRANSFORMATION"
    INVALID = "INVALID"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class MDEValidationResult:
    """
    Structural/type validation result for a (source, candidate) pair.
    This is deliberately independent of the LLM: it must be computable from
    USchemaAttribute/Table/Column metadata alone (see MDEValidator).
    """
    status: MDEStatus
    type_score: float
    structural_score: float
    key_score: float
    cardinality_score: float
    violations: List[str] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# STEP 6.1 — Open-world relevance (for the EVOLVE branch)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelevanceResult:
    """
    Whether a source element with NO valid candidate represents a genuinely
    new, meaningful concept (→ EVOLVE) or an irrelevant/technical field
    (→ REJECT). See RelevancePolicy — this is intentionally a simple,
    documented heuristic for this phase, not a trained classifier.
    """
    relevant: bool
    confidence: float
    rationale: str = ""


# ---------------------------------------------------------------------------
# STEP 6.2 — Final Decision Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionResult:
    """Final, fully-traceable decision for one source element."""
    source_element: str
    selected_candidate: Optional[CandidateResult]
    semantic_decision: Optional[SemanticDecision]
    mde_validation: Optional[MDEValidationResult]
    decision_type: DecisionType
    confidence: float
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSONL-serializable representation, matching the shape in the brief."""
        return {
            "source": self.source_element,
            "candidate": self.selected_candidate.target_element if self.selected_candidate else None,
            "decision": self.decision_type.value,
            "embedding_score": self.semantic_decision.embedding_score if self.semantic_decision else None,
            "llm_score": self.semantic_decision.llm_score if self.semantic_decision else None,
            "combined_score": self.semantic_decision.combined_score if self.semantic_decision else None,
            "type_score": self.mde_validation.type_score if self.mde_validation else None,
            "structure_score": self.mde_validation.structural_score if self.mde_validation else None,
            "mde_status": self.mde_validation.status.value if self.mde_validation else None,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }