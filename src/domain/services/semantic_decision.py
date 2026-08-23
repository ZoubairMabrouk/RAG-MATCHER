"""
SemanticDecisionService — STEP 4.2.

Orchestration layer only: CandidateResult[] -> SemanticDecision. Reuses
AdvancedRAGRetriever (embeddings/FAISS), HybridScoringSystem (score
combination) and, optionally, LLMOrchestrator (LLM scoring) exactly as they
exist today. Does NOT reimplement any of those.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from src.domain.entities.experimental import CandidateResult, SemanticDecision
from src.domain.entities.rag_schema import RetrievalQuery, SourceField
from src.infrastructure.rag.llm_orchestrator import LLMOrchestrator
from src.infrastructure.rag.retriever import AdvancedRAGRetriever
from src.infrastructure.rag.scoring_system import HybridScoringSystem


def candidate_results_from_retrieval(
    source_element: str,
    retrieved: List[Tuple[object, float, float]],
    retrieval_time_ms: float,
) -> List[CandidateResult]:
    """
    STEP 3 glue: convert AdvancedRAGRetriever.retrieve_candidates() output
    (List[Tuple[KnowledgeBaseDocument, bi_score, cross_score]]) into a ranked
    CandidateResult list, WITHOUT discarding anything past Top-1.

    similarity uses the cross-encoder score when available (it is the more
    refined signal after reranking), falling back to the bi-encoder score.
    Both remain accessible via `extra` for full traceability.
    """
    results: List[CandidateResult] = []
    for rank, (doc, bi_score, cross_score) in enumerate(retrieved, start=1):
        target = f"{getattr(doc, 'table', '?')}.{getattr(doc, 'column', '?')}"
        similarity = cross_score if cross_score is not None else bi_score
        results.append(
            CandidateResult(
                source_element=source_element,
                target_element=target,
                rank=rank,
                similarity=float(similarity),
                retrieval_time_ms=retrieval_time_ms,
                extra={"bi_encoder_score": float(bi_score), "cross_encoder_score": float(cross_score)},
            )
        )
    return results


class SemanticDecisionService:
    """
    Single Responsibility: given a source field, retrieve candidates (via
    AdvancedRAGRetriever), keep the full ranked list as CandidateResult[],
    and produce a SemanticDecision for the best candidate using
    HybridScoringSystem (+ optional LLMOrchestrator for the LLM score term).
    """

    def __init__(
        self,
        retriever: AdvancedRAGRetriever,
        scoring_system: HybridScoringSystem,
        llm_orchestrator: Optional[LLMOrchestrator] = None,
    ):
        self._retriever = retriever
        self._scoring_system = scoring_system
        self._llm_orchestrator = llm_orchestrator

    def evaluate(
        self, source_field: SourceField, top_k: int = 10
    ) -> Tuple[List[CandidateResult], Optional[SemanticDecision]]:
        """
        Returns (all_candidates_ranked, best_semantic_decision_or_None).
        """
        query = RetrievalQuery(source_field=source_field, top_k=top_k)

        start = time.time()
        retrieved = self._retriever.retrieve_candidates(query)
        retrieval_time_ms = (time.time() - start) * 1000

        candidates = candidate_results_from_retrieval(source_field.path, retrieved, retrieval_time_ms)

        if not candidates:
            return candidates, None

        # Use the LLM only for the best (rank-1) candidate's score by default,
        # to keep this service cheap; callers wanting LLM scores for every
        # candidate can call _llm_score_for per-candidate themselves.
        best_doc, best_bi, best_cross = retrieved[0]
        llm_score = self._llm_score_for(source_field, best_doc) if self._llm_orchestrator else best_cross

        hybrid_score, feature_scores = self._scoring_system.compute_hybrid_score(
            source_field=source_field,
            candidate_doc=best_doc,
            bi_encoder_score=best_bi,
            cross_encoder_score=best_cross,
            llm_confidence=llm_score,
        )

        decision = SemanticDecision(
            candidate=candidates[0],
            embedding_score=float(best_bi),
            llm_score=float(llm_score),
            combined_score=float(hybrid_score),
            rationale=(
                f"Hybrid score from bi_encoder={best_bi:.3f}, cross_encoder={best_cross:.3f}, "
                f"llm={llm_score:.3f}"
            ),
            extra={"feature_scores": feature_scores},
        )

        return candidates, decision

    def _llm_score_for(self, source_field: SourceField, doc) -> float:
        """
        Best-effort single-candidate LLM confidence. Falls back to 0.0 (not a
        fabricated high score) if the LLM call fails -- callers should treat
        a 0.0 llm_score with a live llm_orchestrator as "LLM unavailable/failed"
        and rely on embedding_score instead; this is surfaced via `extra` in
        the resulting SemanticDecision.
        """
        try:
            result = self._llm_orchestrator.match_field(
                RetrievalQuery(source_field=source_field, top_k=1),
                candidates=[doc],
                bi_scores=[1.0],
                cross_scores=[1.0],
            )
            if result.candidates:
                return float(result.candidates[0].confidence_llm)
            return 0.0
        except Exception:
            return 0.0