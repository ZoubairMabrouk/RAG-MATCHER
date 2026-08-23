from src.domain.entities.experimental import (
    CandidateResult,
    DecisionType,
    MDEStatus,
    MDEValidationResult,
    SemanticDecision,
)
from src.domain.services.decision_policy import DecisionPolicy, DecisionThresholds
from src.domain.services.relevance_policy import RelevancePolicy


def make_semantic_decision(score, target="tbl.col"):
    candidate = CandidateResult("src.field", target, 1, score, 1.0)
    return SemanticDecision(candidate=candidate, embedding_score=score, llm_score=score, combined_score=score)


def make_mde(status, type_score=1.0, structural_score=1.0, key_score=1.0, cardinality_score=1.0):
    return MDEValidationResult(
        status=status,
        type_score=type_score,
        structural_score=structural_score,
        key_score=key_score,
        cardinality_score=cardinality_score,
    )


def test_match_when_score_high_and_mde_valid():
    policy = DecisionPolicy(DecisionThresholds(tau_match=0.85, tau_review=0.70))
    result = policy.decide("src.field", make_semantic_decision(0.9), make_mde(MDEStatus.VALID))
    assert result.decision_type == DecisionType.MATCH


def test_match_when_valid_with_transformation():
    policy = DecisionPolicy(DecisionThresholds(tau_match=0.85, tau_review=0.70))
    result = policy.decide("src.field", make_semantic_decision(0.9), make_mde(MDEStatus.VALID_WITH_TRANSFORMATION))
    assert result.decision_type == DecisionType.MATCH


def test_llm_match_but_mde_invalid_never_produces_match():
    """The mission's hard rule: LLM MATCH + MDE INVALID must NOT be MATCH."""
    policy = DecisionPolicy(DecisionThresholds(tau_match=0.85, tau_review=0.70))
    result = policy.decide("src.field", make_semantic_decision(0.99), make_mde(MDEStatus.INVALID))
    assert result.decision_type != DecisionType.MATCH
    assert result.decision_type == DecisionType.REVIEW


def test_llm_match_but_mde_invalid_forces_reject_when_configured():
    policy = DecisionPolicy(DecisionThresholds(tau_match=0.85, tau_review=0.70, invalid_type_forces_reject=True))
    result = policy.decide("src.field", make_semantic_decision(0.99), make_mde(MDEStatus.INVALID))
    assert result.decision_type == DecisionType.REJECT


def test_review_band_score():
    policy = DecisionPolicy(DecisionThresholds(tau_match=0.85, tau_review=0.70))
    result = policy.decide("src.field", make_semantic_decision(0.75), make_mde(MDEStatus.VALID))
    assert result.decision_type == DecisionType.REVIEW


def test_no_candidate_relevant_evolves():
    policy = DecisionPolicy(
        DecisionThresholds(tau_match=0.85, tau_review=0.70),
        relevance_policy=RelevancePolicy(min_semantic_evidence=0.3, ambiguous_band=0.05),
    )
    result = policy.decide("patient.new_clinical_score", None, None, semantic_evidence_for_relevance=0.9)
    assert result.decision_type == DecisionType.EVOLVE


def test_no_candidate_irrelevant_rejects():
    policy = DecisionPolicy(
        DecisionThresholds(tau_match=0.85, tau_review=0.70),
        relevance_policy=RelevancePolicy(min_semantic_evidence=0.3, ambiguous_band=0.05),
    )
    result = policy.decide("patient._internal_cache", None, None, semantic_evidence_for_relevance=0.9)
    assert result.decision_type == DecisionType.REJECT


def test_low_score_candidate_routes_through_relevance_not_auto_evolve():
    """A weak candidate below tau_review must go through RelevancePolicy,
    never straight to EVOLVE just because 'no valid candidate'."""
    policy = DecisionPolicy(
        DecisionThresholds(tau_match=0.85, tau_review=0.70),
        relevance_policy=RelevancePolicy(min_semantic_evidence=0.3, ambiguous_band=0.05),
    )
    weak = make_semantic_decision(0.2)
    result = policy.decide("patient._internal_debug", weak, make_mde(MDEStatus.INVALID),
                            semantic_evidence_for_relevance=0.1)
    assert result.decision_type == DecisionType.REJECT  # technical pattern wins