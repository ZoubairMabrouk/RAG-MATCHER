"""
SemanticDecisionService (STEP 3/4.2 of the experimental pipeline).

Connects three EXISTING classes end-to-end, without reimplementing any of
their logic:

    AdvancedRAGRetriever  -> raw (doc, bi_score, cross_score) tuples
    LLMOrchestrator        -> per-target LLM confidence + rationale
    HybridScoringSystem     -> combined_score for each candidate

Output: one SemanticDecision per retrieved candidate (NOT just Top-1), so
that downstream MDEValidator / RelevancePolicy / DecisionPolicy can inspect
the full ranked list if needed, and so retrieval logging (PHASE 12) has
everything it needs.

No new abstractions are introduced beyond what is required to glue the
three existing services together and to convert AdvancedRAGRetriever's raw
tuples into CandidateResult (STEP 3 entity).
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple

from src.domain.entities.experimental import CandidateResult, SemanticDecision
from src.domain.entities.rag_schema import RetrievalQuery, SourceField

logger = logging.getLogger(__name__)


def candidate_results_from_retrieval(
    source_element: str,
    retrieved: List[Tuple[Any, float, float]],
    retrieval_time_ms: float = 0.0,
) -> List[CandidateResult]:
    """
    Convert AdvancedRAGRetriever.retrieve_candidates() raw output
    (List[Tuple[KnowledgeBaseDocument, bi_encoder_score, cross_encoder_score]])
    into CandidateResult entities, preserving rank order and ALL candidates
    (PHASE 12 requires the full list, not just Top-1).

    similarity uses the cross-encoder score when present (it is the
    reranked, more reliable signal); the bi-encoder score is kept in
    `extra` for traceability/metrics (Hits@K, MRR use whichever the caller
    wants, so both are preserved).
    """
    results: List[CandidateResult] = []

    for rank, (doc, bi_score, cross_score) in enumerate(
        retrieved,
        start=1,
    ):
        target_element = f"{doc.table}.{doc.column}"

        results.append(
            CandidateResult(
                source_element=source_element,
                target_element=target_element,
                rank=rank,
                similarity=float(cross_score),
                retrieval_time_ms=retrieval_time_ms,
                extra={
                    "bi_encoder_score": float(bi_score),
                    "cross_encoder_score": float(cross_score),
                },
            )
        )

    return results


class SemanticDecisionService:
    """
    Single Responsibility: turn one SourceField into a ranked list of
    SemanticDecision, by orchestrating (never reimplementing) the retriever,
    the LLM orchestrator, and the hybrid scoring system.
    """

    def __init__(
        self,
        retriever,
        llm_orchestrator,
        scoring_system,
        top_k: int = 10,
    ):
        self._retriever = retriever
        self._llm_orchestrator = llm_orchestrator
        self._scoring_system = scoring_system
        self._top_k = top_k

    def evaluate_source_element(
        self,
        source_field: SourceField,
    ) -> List[SemanticDecision]:
        """
        Returns [] if the retriever finds no candidates (case 3: aucun
        candidat). Otherwise returns one SemanticDecision per retrieved
        candidate, ordered by retrieval rank (case 1/2: plusieurs
        candidats / Top-1 is decisions[0]).
        """
        start = time.time()

        query = RetrievalQuery(
            source_field=source_field,
            top_k=self._top_k,
        )

        retrieved = self._retriever.retrieve_candidates(query)

        if not retrieved:
            logger.info(
                "No candidates retrieved for %s",
                source_field.path,
            )
            return []

        retrieval_time_ms = (time.time() - start) * 1000

        candidates = candidate_results_from_retrieval(
            source_field.path,
            retrieved,
            retrieval_time_ms=retrieval_time_ms,
        )

        docs = [doc for doc, _, _ in retrieved]
        bi_scores = [bi for _, bi, _ in retrieved]
        cross_scores = [cross for _, _, cross in retrieved]

        llm_by_target = self._call_llm_safely(
            query,
            docs,
            bi_scores,
            cross_scores,
        )

        decisions: List[SemanticDecision] = []

        for candidate, doc in zip(candidates, docs):
            decisions.append(
                self._build_decision(
                    source_field,
                    candidate,
                    doc,
                    llm_by_target,
                )
            )

        return decisions

    # -- internals ---------------------------------------------------------

    def _call_llm_safely(
        self,
        query,
        docs,
        bi_scores,
        cross_scores,
    ) -> dict:
        """
        Case 6 (LLM response invalide) / case 7 (fallback existant du
        projet): if LLMOrchestrator.match_field raises, or returns something
        without a usable `.candidates` list, we do not crash the whole
        evaluation — we fall back to embedding-only scoring for every
        candidate. This mirrors LLMOrchestrator's own internal
        `_create_error_result` fallback pattern, one level up.
        """
        try:
            match_result = self._llm_orchestrator.match_field(
                query,
                docs,
                bi_scores,
                cross_scores,
            )

        except Exception as exc:  # noqa: BLE001 - deliberate broad catch, logged
            logger.warning(
                "LLMOrchestrator.match_field failed, falling back: %s",
                exc,
            )
            return {}

        candidates = getattr(
            match_result,
            "candidates",
            None,
        )

        if not candidates:
            logger.warning(
                "LLMOrchestrator returned no usable candidates, falling back"
            )
            return {}

        return {
            c.target: c
            for c in candidates
        }

    def _build_decision(
        self,
        source_field: SourceField,
        candidate: CandidateResult,
        doc: Any,
        llm_by_target: dict,
    ) -> SemanticDecision:
        llm_match = llm_by_target.get(
            candidate.target_element
        )

        if llm_match is not None:
            llm_score = float(
                llm_match.confidence_llm
            )
            base_rationale = llm_match.rationale

        else:
            # Fallback: no LLM signal for this candidate (either the LLM
            # call failed entirely, or the LLM's response didn't mention
            # this particular target). Score on embedding signal only.
            llm_score = 0.0
            base_rationale = (
                "llm_unavailable_fallback_embedding_only"
            )

        combined_score, _feature_scores = (
            self._scoring_system.compute_hybrid_score(
                source_field=source_field,
                candidate_doc=doc,
                bi_encoder_score=candidate.extra.get(
                    "bi_encoder_score",
                    candidate.similarity,
                ),
                cross_encoder_score=candidate.similarity,
                llm_confidence=llm_score,
            )
        )

        return SemanticDecision(
            candidate=candidate,
            embedding_score=candidate.similarity,
            llm_score=llm_score,
            combined_score=combined_score,
            rationale=base_rationale,
        )