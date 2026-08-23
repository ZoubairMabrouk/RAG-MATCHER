from types import SimpleNamespace

from src.infrastructure.rag.semantic_decision_service import candidate_results_from_retrieval


def doc(table, column):
    return SimpleNamespace(table=table, column=column)


def test_ordering_and_ranks_preserved():
    retrieved = [
        (doc("t", "a"), 0.9, 0.95),
        (doc("t", "b"), 0.8, 0.85),
        (doc("t", "c"), 0.7, 0.75),
    ]
    results = candidate_results_from_retrieval("src.field", retrieved, retrieval_time_ms=12.0)

    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.target_element for r in results] == ["t.a", "t.b", "t.c"]


def test_top_k_correct_length():
    retrieved = [(doc("t", str(i)), 0.5, 0.5) for i in range(5)]
    results = candidate_results_from_retrieval("src.field", retrieved, retrieval_time_ms=1.0)
    assert len(results) == 5


def test_similarity_conserved_uses_cross_encoder_when_present():
    retrieved = [(doc("t", "a"), 0.6, 0.8)]
    results = candidate_results_from_retrieval("src.field", retrieved, retrieval_time_ms=1.0)
    assert results[0].similarity == 0.8
    assert results[0].extra["bi_encoder_score"] == 0.6
    assert results[0].extra["cross_encoder_score"] == 0.8


def test_retrieval_time_ms_populated():
    retrieved = [(doc("t", "a"), 0.6, 0.8)]
    results = candidate_results_from_retrieval("src.field", retrieved, retrieval_time_ms=42.5)
    assert results[0].retrieval_time_ms == 42.5


def test_no_candidates_returns_empty_list():
    results = candidate_results_from_retrieval("src.field", [], retrieval_time_ms=0.0)
    assert results == []