"""
DecisionPolicy — STEP 6.

Combines SemanticDecision + MDEValidationResult + configurable thresholds
into the final DecisionType (MATCH/EVOLVE/REVIEW/REJECT).

Hard rule enforced here (mission, "RÈGLE TRÈS IMPORTANTE SUR LE LLM"):
LLM says MATCH + MDE says INVALID must NEVER produce MATCH. This class is
the one place that rule is implemented; SemanticDecisionService and
MDEValidator are deliberately unaware of each other's conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.domain.entities.experimental import (
    DecisionResult,
    DecisionType,
    MDEStatus,
    MDEValidationResult,
    RelevanceResult,
    SemanticDecision,
)
from src.domain.services.relevance_policy import RelevancePolicy


@dataclass
class DecisionThresholds:
    """Configurable thresholds -- never optimized on test data (see mission
    rule on thresholds); must be recorded in run_config.json by the caller."""
    tau_match: float = 0.85
    tau_review: float = 0.70
    invalid_type_forces_reject: bool = False
    """If True, an MDE INVALID status always forces REJECT (never REVIEW)
    even when combined_score is high. If False (default), INVALID at
    review-eligible score becomes REVIEW instead of an outright REJECT,
    giving a human a chance to catch a genuinely new relationship the
    validator's limited structural model missed."""


class DecisionPolicy:
    """
    Single Responsibility: turn (SemanticDecision, MDEValidationResult,
    RelevanceResult-or-None, thresholds) into one DecisionResult.
    Does not perform retrieval, scoring, or structural validation itself.
    """

    def __init__(
        self,
        thresholds: Optional[DecisionThresholds] = None,
        relevance_policy: Optional[RelevancePolicy] = None,
    ):
        self._thresholds = thresholds or DecisionThresholds()
        self._relevance_policy = relevance_policy or RelevancePolicy()

    def decide(
        self,
        source_element: str,
        semantic_decision: Optional[SemanticDecision],
        mde_validation: Optional[MDEValidationResult],
        semantic_evidence_for_relevance: float = 0.0,
    ) -> DecisionResult:
        """
        Args:
            source_element: dotted path of the source attribute.
            semantic_decision: best semantic decision found for this source,
                or None if retrieval produced no candidates at all.
            mde_validation: MDE result for the (source, best-candidate) pair,
                or None if there was no candidate to validate.
            semantic_evidence_for_relevance: signal fed to RelevancePolicy
                when there is no valid candidate at all (see STEP 6.1).
        """
        has_candidate = semantic_decision is not None and semantic_decision.candidate is not None

        if not has_candidate:
            return self._decide_no_candidate(source_element, semantic_evidence_for_relevance)

        score = semantic_decision.combined_score
        mde_status = mde_validation.status if mde_validation else None

        # --- MATCH branch ---
        if score >= self._thresholds.tau_match and mde_status == MDEStatus.VALID:
            return self._build(source_element, semantic_decision, mde_validation, DecisionType.MATCH, score,
                                "Score above tau_match and MDE validation is VALID.")

        # Also allow VALID_WITH_TRANSFORMATION into MATCH: it is a compatible
        # match that merely needs a type cast, not a structural problem.
        if score >= self._thresholds.tau_match and mde_status == MDEStatus.VALID_WITH_TRANSFORMATION:
            return self._build(source_element, semantic_decision, mde_validation, DecisionType.MATCH, score,
                                "Score above tau_match; MDE validation compatible with transformation.")

        # --- Explicit hard rule: LLM/semantic MATCH-level score but MDE says
        # INVALID must NEVER become MATCH. ---
        if score >= self._thresholds.tau_match and mde_status == MDEStatus.INVALID:
            if self._thresholds.invalid_type_forces_reject:
                return self._build(source_element, semantic_decision, mde_validation, DecisionType.REJECT, score,
                                    "High semantic score but MDE INVALID; policy forces REJECT.")
            return self._build(source_element, semantic_decision, mde_validation, DecisionType.REVIEW, score,
                                "High semantic score but MDE INVALID; downgraded to REVIEW "
                                "(LLM is not the structural authority).")

        # --- REVIEW branch (score in the review band) ---
        if score >= self._thresholds.tau_review:
            if mde_status in (MDEStatus.VALID, MDEStatus.VALID_WITH_TRANSFORMATION, MDEStatus.REVIEW, None):
                return self._build(source_element, semantic_decision, mde_validation, DecisionType.REVIEW, score,
                                    "Score in review band.")
            if mde_status == MDEStatus.INVALID:
                if self._thresholds.invalid_type_forces_reject:
                    return self._build(source_element, semantic_decision, mde_validation, DecisionType.REJECT, score,
                                        "Score in review band but MDE INVALID; policy forces REJECT.")
                return self._build(source_element, semantic_decision, mde_validation, DecisionType.REVIEW, score,
                                    "Score in review band and MDE INVALID; kept as REVIEW.")

        # --- below tau_review with a candidate: treat as "no valid candidate" ---
        return self._decide_no_candidate(source_element, semantic_evidence_for_relevance,
                                          semantic_decision=semantic_decision, mde_validation=mde_validation)

    # ------------------------------------------------------------------

    def _decide_no_candidate(
        self,
        source_element: str,
        semantic_evidence: float,
        semantic_decision: Optional[SemanticDecision] = None,
        mde_validation: Optional[MDEValidationResult] = None,
    ) -> DecisionResult:
        relevance: RelevanceResult = self._relevance_policy.assess(source_element, semantic_evidence)

        if self._relevance_policy.is_ambiguous(relevance):
            decision_type = DecisionType.REVIEW
        elif relevance.relevant:
            decision_type = DecisionType.EVOLVE
        else:
            decision_type = DecisionType.REJECT

        return self._build(
            source_element,
            semantic_decision,
            mde_validation,
            decision_type,
            relevance.confidence,
            relevance.rationale,
        )

    def _build(
        self,
        source_element: str,
        semantic_decision: Optional[SemanticDecision],
        mde_validation: Optional[MDEValidationResult],
        decision_type: DecisionType,
        confidence: float,
        rationale: str,
    ) -> DecisionResult:
        candidate = semantic_decision.candidate if semantic_decision else None
        return DecisionResult(
            source_element=source_element,
            selected_candidate=candidate,
            semantic_decision=semantic_decision,
            mde_validation=mde_validation,
            decision_type=decision_type,
            confidence=confidence,
            rationale=rationale,
        )