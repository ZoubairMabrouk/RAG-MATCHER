"""
Main RAG Orchestrator for schema matching.
Coordinates all RAG components for end-to-end matching.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.domain.entities.rag_schema import (
    SourceField, RetrievalQuery, SchemaMatchingResult, 
    ScoringWeights, ScoringThresholds, FieldType
)
from src.infrastructure.rag.embedding_service import RAGEmbeddingService
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.retriever import AdvancedRAGRetriever
from src.infrastructure.rag.llm_orchestrator import LLMOrchestrator
from src.infrastructure.rag.scoring_system import HybridScoringSystem
from src.infrastructure.llm.llm_client import ILLMClient

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Main orchestrator for RAG-based schema matching.
    Coordinates retrieval, LLM processing, and scoring.
    """
    
    def __init__(self, 
                 vector_store: RAGVectorStore,
                 embedding_service: RAGEmbeddingService,
                 llm_client: ILLMClient,
                 weights: Optional[ScoringWeights] = None,
                 thresholds: Optional[ScoringThresholds] = None):
        """
        Initialize RAG orchestrator with all components.
        
        Args:
            vector_store: Vector store for document retrieval
            embedding_service: Embedding service for queries and documents
            llm_client: LLM client for generation
            weights: Scoring weights configuration
            thresholds: Decision thresholds configuration
        """
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._llm_client = llm_client
        
        # Initialize components
        self._retriever = AdvancedRAGRetriever(vector_store, embedding_service)
        self._llm_orchestrator = LLMOrchestrator(llm_client, weights, thresholds)
        self._scoring_system = HybridScoringSystem(weights, thresholds)
        
        logger.info("RAG Orchestrator initialized")
    
    def match_schema_fields(self, source_fields: List[SourceField], 
                          top_k: int = 20) -> List[SchemaMatchingResult]:
        """
        Match multiple source fields to MIMIC-III schema.
        
        Args:
            source_fields: List of source fields to match
            top_k: Number of candidates to retrieve per field
            
        Returns:
            List of matching results
        """
        logger.info(f"Starting schema matching for {len(source_fields)} fields")
        start_time = time.time()
        
        results = []
        
        for i, source_field in enumerate(source_fields):
            try:
                logger.info(f"Processing field {i+1}/{len(source_fields)}: {source_field.path}")
                
                # Step 1: Retrieve candidates
                query = RetrievalQuery(source_field=source_field, top_k=top_k)
                candidates_with_scores = self._retriever.retrieve_candidates(query)
                
                if not candidates_with_scores:
                    logger.warning(f"No candidates found for {source_field.path}")
                    result = self._create_no_candidates_result(source_field)
                    results.append(result)
                    continue
                
                # Step 2: Extract documents and scores
                candidate_docs = [doc for doc, _, _ in candidates_with_scores]
                bi_scores = [bi_score for _, bi_score, _ in candidates_with_scores]
                cross_scores = [cross_score for _, _, cross_score in candidates_with_scores]
                
                # Step 3: LLM processing
                llm_result = self._llm_orchestrator.match_field(
                    query, candidate_docs, bi_scores, cross_scores
                )
                
                # Step 4: Apply hybrid scoring
                final_result = self._apply_hybrid_scoring(
                    source_field, llm_result, candidate_docs, bi_scores, cross_scores
                )
                
                results.append(final_result)
                
            except Exception as e:
                logger.error(f"Error processing field {source_field.path}: {e}")
                error_result = self._create_error_result(source_field, str(e))
                results.append(error_result)
        
        total_time = time.time() - start_time
        logger.info(f"Schema matching completed in {total_time:.2f}s for {len(source_fields)} fields")
        
        return results
    
    def match_single_field(self, source_field: SourceField, 
                         top_k: int = 20) -> SchemaMatchingResult:
        """
        Match a single source field to MIMIC-III schema.
        
        Args:
            source_field: Source field to match
            top_k: Number of candidates to retrieve
            
        Returns:
            Matching result
        """
        return self.match_schema_fields([source_field], top_k)[0]
    
    def _apply_hybrid_scoring(self, 
                            source_field: SourceField,
                            llm_result: SchemaMatchingResult,
                            candidate_docs: List,
                            bi_scores: List[float],
                            cross_scores: List[float]) -> SchemaMatchingResult:
        """Apply hybrid scoring to LLM result."""
        # Compute additional feature scores for each candidate
        feature_scores_list = []
        enhanced_candidates = []
        
        for i, (doc, bi_score, cross_score) in enumerate(zip(candidate_docs, bi_scores, cross_scores)):
            # Get LLM confidence for this candidate
            llm_confidence = 0.5  # Default
            for candidate in llm_result.candidates:
                if candidate.target == f"{doc.table}.{doc.column}":
                    llm_confidence = candidate.confidence_llm
                    break
            
            # Compute hybrid score
            hybrid_score, feature_scores = self._scoring_system.compute_hybrid_score(
                source_field, doc, bi_score, cross_score, llm_confidence
            )
            
            feature_scores_list.append(feature_scores)
            
            # Create enhanced candidate
            enhanced_candidate = type(llm_result.candidates[0])(
                target=f"{doc.table}.{doc.column}",
                confidence_model=hybrid_score,
                confidence_llm=llm_confidence,
                rationale=next((c.rationale for c in llm_result.candidates 
                              if c.target == f"{doc.table}.{doc.column}"), "No rationale"),
                guardrails=[]
            )
            enhanced_candidates.append(enhanced_candidate)
        
        # Make final decision with hybrid scoring
        final_decision = self._scoring_system.make_decision(
            source_field, enhanced_candidates, feature_scores_list
        )
        
        # Create final result
        return SchemaMatchingResult(
            source_field=llm_result.source_field,
            candidates=enhanced_candidates,
            decision=final_decision,
            processing_time_ms=llm_result.processing_time_ms,
            model_version=llm_result.model_version
        )
    
    def _create_no_candidates_result(self, source_field: SourceField) -> SchemaMatchingResult:
        """Create result when no candidates are found."""
        from src.domain.entities.rag_schema import MatchingDecision, DecisionAction
        
        decision = MatchingDecision(
            action=DecisionAction.REJECT,
            selected_target=None,
            final_confidence=0.0,
            guardrails=["no_candidates_found"],
            review_checklist=["Vérifier si le champ existe dans le schéma source"]
        )
        
        return SchemaMatchingResult(
            source_field=source_field.path,
            candidates=[],
            decision=decision,
            processing_time_ms=0.0,
            model_version="1.0"
        )
    
    def _create_error_result(self, source_field: SourceField, error_msg: str) -> SchemaMatchingResult:
        """Create result when processing fails."""
        from src.domain.entities.rag_schema import MatchingDecision, DecisionAction
        
        decision = MatchingDecision(
            action=DecisionAction.REJECT,
            selected_target=None,
            final_confidence=0.0,
            guardrails=["processing_error"],
            review_checklist=[f"Erreur de traitement: {error_msg}"]
        )
        
        return SchemaMatchingResult(
            source_field=source_field.path,
            candidates=[],
            decision=decision,
            processing_time_ms=0.0,
            model_version="1.0"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        vector_stats = self._vector_store.get_statistics()
        
        return {
            "vector_store": vector_stats,
            "embedding_dimension": self._embedding_service.get_embedding_dimension(),
            "scoring_weights": self._scoring_system._weights.dict(),
            "decision_thresholds": self._scoring_system._thresholds.dict(),
            "is_calibrated": self._scoring_system._is_calibrated
        }
    
    def calibrate_system(self, training_data: List[Dict[str, Any]]) -> None:
        """Calibrate the scoring system with training data."""
        logger.info(f"Calibrating system with {len(training_data)} training examples")
        self._scoring_system.calibrate(training_data)
    
    def update_scoring_config(self, 
                            weights: Optional[ScoringWeights] = None,
                            thresholds: Optional[ScoringThresholds] = None) -> None:
        """Update scoring configuration."""
        if weights:
            self._scoring_system.update_weights(weights)
            self._llm_orchestrator._weights = weights
        
        if thresholds:
            self._scoring_system.update_thresholds(thresholds)
            self._llm_orchestrator._thresholds = thresholds
        
        logger.info("Updated scoring configuration")


class RAGService:
    """
    High-level service for RAG-based schema matching.
    Provides simplified interface for common operations.
    """
    
    def __init__(self, orchestrator: RAGOrchestrator):
        self._orchestrator = orchestrator
    
    def match_fields_from_json(self, source_fields_json: List[Dict[str, Any]], 
                             top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Match fields from JSON representation.
        
        Args:
            source_fields_json: List of field dictionaries
            top_k: Number of candidates to retrieve
            
        Returns:
            List of matching results as dictionaries
        """
        # Convert JSON to SourceField objects
        source_fields = []
        for field_json in source_fields_json:
            try:
                source_field = SourceField(**field_json)
                source_fields.append(source_field)
            except Exception as e:
                logger.error(f"Error parsing field: {e}")
                continue
        
        # Perform matching
        results = self._orchestrator.match_schema_fields(source_fields, top_k)
        
        # Convert results to dictionaries
        return [result.dict() for result in results]
    
    def match_field_by_path(self, field_path: str, 
                          name_tokens: List[str],
                          inferred_type: str,
                          **kwargs) -> Dict[str, Any]:
        """
        Match a single field by path and basic information.
        
        Args:
            field_path: Path to the field
            name_tokens: Tokenized field name
            inferred_type: Inferred data type
            **kwargs: Additional field properties
            
        Returns:
            Matching result as dictionary
        """
        # Create SourceField
        source_field = SourceField(
            path=field_path,
            name_tokens=name_tokens,
            inferred_type=FieldType(inferred_type),
            **kwargs
        )
        
        # Perform matching
        result = self._orchestrator.match_single_field(source_field)
        
        return result.dict()
    
    def batch_match_fields(self, field_batch: List[Dict[str, Any]], 
                         batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Match fields in batches for better performance.
        
        Args:
            field_batch: List of field dictionaries
            batch_size: Size of each batch
            
        Returns:
            List of matching results
        """
        all_results = []
        
        for i in range(0, len(field_batch), batch_size):
            batch = field_batch[i:i + batch_size]
            batch_results = self.match_fields_from_json(batch)
            all_results.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(field_batch)-1)//batch_size + 1}")
        
        return all_results
    
    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get matching statistics."""
        return self._orchestrator.get_statistics()
    
    def search_semantic_hints(self, hints: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search by semantic hints without specific field context.
        
        Args:
            hints: List of semantic hints
            top_k: Number of results to return
            
        Returns:
            List of matching documents
        """
        documents = self._orchestrator._retriever.search_by_semantics(hints, top_k)
        return [doc.dict() for doc in documents]

