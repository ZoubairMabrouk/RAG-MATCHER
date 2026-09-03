"""
STEP 3/4.2 tests for SemanticDecisionService.

No real OpenAI / Ollama / FAISS / PostgreSQL — AdvancedRAGRetriever,
LLMOrchestrator and HybridScoringSystem are replaced with lightweight
doubles that expose only the methods SemanticDecisionService calls.

Covers the 7 required cases:
  1. plusieurs candidats
  2. candidat Top-1
  3. aucun candidat
  4. score eleve
  5. score faible
  6. LLM response invalide
  7. fallback existant du projet
"""

from types import SimpleNamespace

import pytest

from src.domain.entities.experimental import SemanticDecision
from src.domain.entities.rag_schema import SourceField, FieldType
from src.infrastructure.rag.semantic_decision_service import (
    SemanticDecisionService,
)


def make_doc(table, column):
    return SimpleNamespace(
        table=table,
        column=column,
    )


def make_source_field(
    path="patient.new_score",
):
    return SourceField(
        path=path,
        name_tokens=path.split("."),
        inferred_type=FieldType.FLOAT,
    )


class FakeRetriever:
    """Double for AdvancedRAGRetriever."""

    def __init__(self, retrieved):
        self._retrieved = retrieved
        self.last_query = None

    def retrieve_candidates(self, query):
        self.last_query = query
        return self._retrieved


class FakeScoringSystem:
    """Double for HybridScoringSystem: deterministic average of the three signals."""

    def compute_hybrid_score(
        self,
        source_field,
        candidate_doc,
        bi_encoder_score,
        cross_encoder_score,
        llm_confidence,
        additional_features=None,
    ):
        score = (
            bi_encoder_score
            + cross_encoder_score
            + llm_confidence
        ) / 3.0

        return score, {
            "bi_encoder": bi_encoder_score,
            "cross_encoder": cross_encoder_score,
            "llm_confidence": llm_confidence,
        }


class FakeLLMOrchestrator:
    """
    Double for LLMOrchestrator.match_field.

    by_target maps 'table.column' -> CandidateMatch-like.
    """

    def __init__(
        self,
        by_target=None,
        raise_exc=None,
        return_invalid=False,
    ):
        self._by_target = by_target or {}
        self._raise_exc = raise_exc
        self._return_invalid = return_invalid
        self.called_with = None

    def match_field(
        self,
        query,
        docs,
        bi_scores,
        cross_scores,
    ):
        self.called_with = (
            query,
            docs,
            bi_scores,
            cross_scores,
        )

        if self._raise_exc is not None:
            raise self._raise_exc

        if self._return_invalid:
            return SimpleNamespace(
                candidates=None
            )

        return SimpleNamespace(
            candidates=list(
                self._by_target.values()
            )
        )


def make_llm_candidate(
    target,
    confidence_llm,
    rationale="ok",
):
    return SimpleNamespace(
        target=target,
        confidence_llm=confidence_llm,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Case 3: aucun candidat
# ---------------------------------------------------------------------------


def test_no_candidates_returns_empty_list():
    service = SemanticDecisionService(
        retriever=FakeRetriever([]),
        llm_orchestrator=FakeLLMOrchestrator(),
        scoring_system=FakeScoringSystem(),
    )

    decisions = service.evaluate_source_element(
        make_source_field()
    )

    assert decisions == []


# ---------------------------------------------------------------------------
# Case 1 + 2: plusieurs candidats, candidat Top-1
# ---------------------------------------------------------------------------


def test_multiple_candidates_ranked_and_top1_first():
    retrieved = [
        (
            make_doc("patient", "score"),
            0.9,
            0.95,
        ),
        (
            make_doc("patient", "value"),
            0.6,
            0.65,
        ),
        (
            make_doc("patient", "note"),
            0.3,
            0.35,
        ),
    ]

    llm_matches = {
        "patient.score": make_llm_candidate(
            "patient.score",
            0.92,
            "strong semantic match",
        ),
        "patient.value": make_llm_candidate(
            "patient.value",
            0.5,
            "partial match",
        ),
        "patient.note": make_llm_candidate(
            "patient.note",
            0.1,
            "unrelated",
        ),
    }

    service = SemanticDecisionService(
        retriever=FakeRetriever(retrieved),
        llm_orchestrator=FakeLLMOrchestrator(
            by_target=llm_matches
        ),
        scoring_system=FakeScoringSystem(),
    )

    decisions = service.evaluate_source_element(
        make_source_field()
    )

    assert len(decisions) == 3
    assert all(
        isinstance(d, SemanticDecision)
        for d in decisions
    )

    top1 = decisions[0]

    assert top1.candidate.rank == 1
    assert (
        top1.candidate.target_element
        == "patient.score"
    )

    assert top1.embedding_score == 0.95
    assert top1.llm_score == 0.92

    assert top1.combined_score == pytest.approx(
        (0.9 + 0.95 + 0.92) / 3.0,
        rel=1e-6,
    )

    assert top1.rationale == "strong semantic match"

    # rank ordering preserved end-to-end
    assert [
        d.candidate.rank
        for d in decisions
    ] == [1, 2, 3]

    assert [
        d.candidate.target_element
        for d in decisions
    ] == [
        "patient.score",
        "patient.value",
        "patient.note",
    ]


# ---------------------------------------------------------------------------
# Case 4: score eleve
# ---------------------------------------------------------------------------


def test_high_score_candidate_values():
    retrieved = [
        (
            make_doc("patient", "score"),
            0.95,
            0.97,
        )
    ]

    llm_matches = {
        "patient.score": make_llm_candidate(
            "patient.score",
            0.98,
            "excellent match",
        )
    }

    service = SemanticDecisionService(
        retriever=FakeRetriever(retrieved),
        llm_orchestrator=FakeLLMOrchestrator(
            by_target=llm_matches
        ),
        scoring_system=FakeScoringSystem(),
    )

    decisions = service.evaluate_source_element(
        make_source_field()
    )

    assert len(decisions) == 1

    d = decisions[0]

    assert d.embedding_score == 0.97
    assert d.llm_score == 0.98
    assert d.combined_score > 0.9
    assert d.rationale == "excellent match"


# ---------------------------------------------------------------------------
# Case 5: score faible
# ---------------------------------------------------------------------------


def test_low_score_candidate_values():
    retrieved = [
        (
            make_doc(
                "patient",
                "internal_debug",
            ),
            0.1,
            0.12,
        )
    ]

    llm_matches = {
        "patient.internal_debug": make_llm_candidate(
            "patient.internal_debug",
            0.05,
            "no relation",
        )
    }

    service = SemanticDecisionService(
        retriever=FakeRetriever(retrieved),
        llm_orchestrator=FakeLLMOrchestrator(
            by_target=llm_matches
        ),
        scoring_system=FakeScoringSystem(),
    )

    decisions = service.evaluate_source_element(
        make_source_field(
            path="patient.internal_debug"
        )
    )

    assert len(decisions) == 1

    d = decisions[0]

    assert d.embedding_score == 0.12
    assert d.llm_score == 0.05
    assert d.combined_score < 0.15
    assert d.rationale == "no relation"


# ---------------------------------------------------------------------------
# Case 6: LLM response invalide
# ---------------------------------------------------------------------------


def test_invalid_llm_response_falls_back_to_embedding_only():
    retrieved = [
        (
            make_doc("patient", "score"),
            0.8,
            0.85,
        )
    ]

    service = SemanticDecisionService(
        retriever=FakeRetriever(retrieved),
        llm_orchestrator=FakeLLMOrchestrator(
            return_invalid=True
        ),
        scoring_system=FakeScoringSystem(),
    )

    decisions = service.evaluate_source_element(
        make_source_field()
    )

    assert len(decisions) == 1

    d = decisions[0]

    assert d.llm_score == 0.0
    assert d.embedding_score == 0.85

    assert (
        d.rationale
        == "llm_unavailable_fallback_embedding_only"
    )

    # combined_score still computed
    # (embedding-only degraded mode), never crashes
    assert d.combined_score == pytest.approx(
        (0.8 + 0.85 + 0.0) / 3.0,
        rel=1e-6,
    )


# ---------------------------------------------------------------------------
# Case 7: fallback existant du projet (LLM call raises)
# ---------------------------------------------------------------------------


def test_llm_exception_triggers_fallback_without_crashing():
    retrieved = [
        (
            make_doc("patient", "score"),
            0.7,
            0.75,
        ),
        (
            make_doc("patient", "value"),
            0.4,
            0.45,
        ),
    ]

    service = SemanticDecisionService(
        retriever=FakeRetriever(retrieved),
        llm_orchestrator=FakeLLMOrchestrator(
            raise_exc=RuntimeError(
                "LLM API down"
            )
        ),
        scoring_system=FakeScoringSystem(),
    )

    decisions = service.evaluate_source_element(
        make_source_field()
    )

    assert len(decisions) == 2

    for d in decisions:
        assert d.llm_score == 0.0
        assert (
            d.rationale
            == "llm_unavailable_fallback_embedding_only"
        )

    assert decisions[0].embedding_score == 0.75
    assert decisions[1].embedding_score == 0.45


# ---------------------------------------------------------------------------
# Traceability: retrieval query built from the correct SourceField / top_k
# ---------------------------------------------------------------------------


def test_query_uses_configured_top_k():
    retriever = FakeRetriever(
        [
            (
                make_doc("patient", "score"),
                0.5,
                0.5,
            )
        ]
    )

    service = SemanticDecisionService(
        retriever=retriever,
        llm_orchestrator=FakeLLMOrchestrator(),
        scoring_system=FakeScoringSystem(),
        top_k=3,
    )

    service.evaluate_source_element(
        make_source_field()
    )

    assert retriever.last_query.top_k == 3