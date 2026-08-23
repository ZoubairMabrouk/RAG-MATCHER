import json

from src.domain.entities.experimental import (
    CandidateResult,
    DecisionResult,
    DecisionType,
    MDEStatus,
    MDEValidationResult,
    RelevanceResult,
    SemanticDecision,
    decision_action_to_type,
)
from src.domain.entities.rag_schema import DecisionAction


def test_decision_type_values():
    assert DecisionType.MATCH.value == "MATCH"
    assert DecisionType.EVOLVE.value == "EVOLVE"
    assert DecisionType.REVIEW.value == "REVIEW"
    assert DecisionType.REJECT.value == "REJECT"


def test_decision_type_serialization_roundtrip():
    dt = DecisionType.EVOLVE
    assert DecisionType(dt.value) == dt
    assert json.dumps({"decision": dt.value}) == '{"decision": "EVOLVE"}'


def test_decision_action_to_type_mapping():
    assert decision_action_to_type(DecisionAction.ACCEPT) == DecisionType.MATCH
    assert decision_action_to_type(DecisionAction.REVIEW) == DecisionType.REVIEW
    assert decision_action_to_type(DecisionAction.REJECT) == DecisionType.REJECT


def test_candidate_result_ranking_fields():
    c = CandidateResult(
        source_element="patient.device_id",
        target_element="devices.device_id",
        rank=1,
        similarity=0.87,
        retrieval_time_ms=12.3,
    )
    assert c.rank == 1
    assert c.similarity == 0.87
    assert c.retrieval_time_ms == 12.3
    assert c.is_gold is None


def test_decision_result_to_dict_shape():
    candidate = CandidateResult("patient.device.id", "devices.device_id", 1, 0.91, 5.0)
    semantic = SemanticDecision(candidate=candidate, embedding_score=0.91, llm_score=0.94, combined_score=0.93)
    mde = MDEValidationResult(
        status=MDEStatus.VALID,
        type_score=1.0,
        structural_score=1.0,
        key_score=1.0,
        cardinality_score=1.0,
    )
    result = DecisionResult(
        source_element="patient.device.id",
        selected_candidate=candidate,
        semantic_decision=semantic,
        mde_validation=mde,
        decision_type=DecisionType.MATCH,
        confidence=0.93,
        rationale="ok",
    )
    d = result.to_dict()
    assert d["source"] == "patient.device.id"
    assert d["candidate"] == "devices.device_id"
    assert d["decision"] == "MATCH"
    assert d["mde_status"] == "VALID"
    assert d["confidence"] == 0.93


def test_relevance_result_fields():
    r = RelevanceResult(relevant=True, confidence=0.8, rationale="test")
    assert r.relevant is True
    assert r.confidence == 0.8