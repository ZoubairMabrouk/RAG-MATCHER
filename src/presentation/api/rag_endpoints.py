"""
RAG API endpoints for schema matching and review.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from src.infrastructure.rag.rag_orchestrator import RAGService
from src.domain.entities.rag_schema import (
    SourceField, FieldType, DecisionAction, 
    ScoringWeights, ScoringThresholds
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/rag", tags=["RAG Schema Matching"])

# Global service instance (would be injected via DI in production)
_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """Dependency to get RAG service instance."""
    if _rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return _rag_service

def set_rag_service(service: RAGService) -> None:
    """Set RAG service instance."""
    global _rag_service
    _rag_service = service


# Request/Response Models
class FieldMatchRequest(BaseModel):
    """Request model for single field matching."""
    path: str = Field(..., description="Field path")
    name_tokens: List[str] = Field(..., description="Tokenized field name")
    inferred_type: str = Field(..., description="Inferred data type")
    format_regex: Optional[str] = Field(None, description="Format pattern")
    units: Optional[str] = Field(None, description="Units")
    category_values: Optional[List[str]] = Field(None, description="Sample values")
    hints: List[str] = Field(default_factory=list, description="Semantic hints")
    neighbors: List[str] = Field(default_factory=list, description="Nearby fields")
    coarse_semantics: List[str] = Field(default_factory=list, description="Semantic categories")


class BatchMatchRequest(BaseModel):
    """Request model for batch field matching."""
    fields: List[FieldMatchRequest] = Field(..., description="List of fields to match")
    top_k: int = Field(20, description="Number of candidates to retrieve")
    batch_size: int = Field(10, description="Batch processing size")


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    hints: List[str] = Field(..., description="Semantic hints")
    top_k: int = Field(10, description="Number of results")


class ScoringConfigRequest(BaseModel):
    """Request model for updating scoring configuration."""
    weights: Optional[ScoringWeights] = Field(None, description="Scoring weights")
    thresholds: Optional[ScoringThresholds] = Field(None, description="Decision thresholds")


class ReviewFeedbackRequest(BaseModel):
    """Request model for human review feedback."""
    source_field: str = Field(..., description="Source field path")
    correct_target: Optional[str] = Field(None, description="Correct target column")
    feedback_type: str = Field(..., description="Type of feedback")
    comments: Optional[str] = Field(None, description="Additional comments")


class CalibrationRequest(BaseModel):
    """Request model for system calibration."""
    training_examples: List[Dict[str, Any]] = Field(..., description="Training examples")


# API Endpoints
@router.post("/match/single")
async def match_single_field(
    request: FieldMatchRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Match a single source field to MIMIC-III schema.
    
    Args:
        request: Field matching request
        rag_service: RAG service instance
        
    Returns:
        Matching result
    """
    try:
        # Convert request to SourceField
        source_field = SourceField(
            path=request.path,
            name_tokens=request.name_tokens,
            inferred_type=FieldType(request.inferred_type),
            format_regex=request.format_regex,
            units=request.units,
            category_values=request.category_values,
            hints=request.hints,
            neighbors=request.neighbors,
            coarse_semantics=request.coarse_semantics
        )
        
        # Perform matching
        result = rag_service._orchestrator.match_single_field(source_field)
        
        return result.dict()
        
    except Exception as e:
        logger.error(f"Error in single field matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match/batch")
async def match_batch_fields(
    request: BatchMatchRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Match multiple source fields to MIMIC-III schema.
    
    Args:
        request: Batch matching request
        background_tasks: Background tasks for async processing
        rag_service: RAG service instance
        
    Returns:
        Batch matching results
    """
    try:
        # Convert requests to SourceField objects
        source_fields = []
        for field_req in request.fields:
            source_field = SourceField(
                path=field_req.path,
                name_tokens=field_req.name_tokens,
                inferred_type=FieldType(field_req.inferred_type),
                format_regex=field_req.format_regex,
                units=field_req.units,
                category_values=field_req.category_values,
                hints=field_req.hints,
                neighbors=field_req.neighbors,
                coarse_semantics=field_req.coarse_semantics
            )
            source_fields.append(source_field)
        
        # Perform matching
        if request.batch_size > 1:
            # Process in batches
            results = rag_service.batch_match_fields(
                [field.dict() for field in request.fields],
                request.batch_size
            )
        else:
            # Process all at once
            results = rag_service.match_fields_from_json(
                [field.dict() for field in request.fields],
                request.top_k
            )
        
        # Calculate statistics
        total_fields = len(results)
        accept_count = sum(1 for r in results if r["decision"]["action"] == "ACCEPT")
        review_count = sum(1 for r in results if r["decision"]["action"] == "REVIEW")
        reject_count = sum(1 for r in results if r["decision"]["action"] == "REJECT")
        
        return {
            "results": results,
            "statistics": {
                "total_fields": total_fields,
                "accept_count": accept_count,
                "review_count": review_count,
                "reject_count": reject_count,
                "accept_rate": accept_count / total_fields if total_fields > 0 else 0,
                "review_rate": review_count / total_fields if total_fields > 0 else 0,
                "reject_rate": reject_count / total_fields if total_fields > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch field matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Search MIMIC-III schema by semantic hints.
    
    Args:
        request: Semantic search request
        rag_service: RAG service instance
        
    Returns:
        Search results
    """
    try:
        results = rag_service.search_semantic_hints(request.hints, request.top_k)
        
        return {
            "hints": request.hints,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get RAG system statistics.
    
    Args:
        rag_service: RAG service instance
        
    Returns:
        System statistics
    """
    try:
        stats = rag_service.get_matching_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config/scoring")
async def update_scoring_config(
    request: ScoringConfigRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, str]:
    """
    Update scoring configuration.
    
    Args:
        request: Scoring configuration request
        rag_service: RAG service instance
        
    Returns:
        Update status
    """
    try:
        rag_service._orchestrator.update_scoring_config(
            weights=request.weights,
            thresholds=request.thresholds
        )
        
        return {"status": "success", "message": "Scoring configuration updated"}
        
    except Exception as e:
        logger.error(f"Error updating scoring config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calibrate")
async def calibrate_system(
    request: CalibrationRequest,
    background_tasks: BackgroundTasks,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, str]:
    """
    Calibrate the scoring system with training data.
    
    Args:
        request: Calibration request
        background_tasks: Background tasks for async processing
        rag_service: RAG service instance
        
    Returns:
        Calibration status
    """
    try:
        # Run calibration in background
        background_tasks.add_task(
            rag_service._orchestrator.calibrate_system,
            request.training_examples
        )
        
        return {
            "status": "accepted",
            "message": f"Calibration started with {len(request.training_examples)} examples"
        }
        
    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/feedback")
async def submit_review_feedback(
    request: ReviewFeedbackRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, str]:
    """
    Submit human review feedback for model improvement.
    
    Args:
        request: Review feedback request
        rag_service: RAG service instance
        
    Returns:
        Feedback submission status
    """
    try:
        # In a real implementation, this would:
        # 1. Store feedback in database
        # 2. Update model parameters
        # 3. Retrain components if needed
        
        logger.info(f"Received feedback for {request.source_field}: {request.feedback_type}")
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy", "service": "RAG Schema Matching"}


# Review-specific endpoints
@router.get("/review/pending")
async def get_pending_reviews(
    limit: int = 50,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get pending reviews for human annotation.
    
    Args:
        limit: Maximum number of reviews to return
        rag_service: RAG service instance
        
    Returns:
        Pending reviews
    """
    try:
        # In a real implementation, this would query a database
        # for items marked as REVIEW that need human attention
        
        return {
            "pending_reviews": [],
            "total_count": 0,
            "message": "No pending reviews found"
        }
        
    except Exception as e:
        logger.error(f"Error getting pending reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/review/statistics")
async def get_review_statistics(
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, Any]:
    """
    Get review statistics.
    
    Args:
        rag_service: RAG service instance
        
    Returns:
        Review statistics
    """
    try:
        # In a real implementation, this would query statistics
        # from the review database
        
        return {
            "total_reviews": 0,
            "completed_reviews": 0,
            "pending_reviews": 0,
            "average_review_time": 0,
            "accuracy_rate": 0.0
        }
        
    except Exception as e:
        logger.error(f"Error getting review statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

