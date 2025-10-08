"""
Hybrid scoring system for RAG-based schema matching.
Combines multiple signals for robust decision making.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.domain.entities.rag_schema import (
    SourceField, KnowledgeBaseDocument, CandidateMatch, 
    MatchingDecision, DecisionAction, ScoringWeights, ScoringThresholds,
    SchemaMatchingResult
)

logger = logging.getLogger(__name__)


class HybridScoringSystem:
    """
    Hybrid scoring system combining multiple signals for robust matching decisions.
    Single Responsibility: Scoring and calibration for matching decisions.
    """
    
    def __init__(self, weights: Optional[ScoringWeights] = None,
                 thresholds: Optional[ScoringThresholds] = None,
                 enable_calibration: bool = True):
        self._weights = weights or ScoringWeights()
        self._thresholds = thresholds or ScoringThresholds()
        self._enable_calibration = enable_calibration
        
        # Calibration models
        self._calibrator = None
        self._is_calibrated = False
        
        # Feature importance tracking
        self._feature_importance = {
            "bi_encoder": 0.0,
            "cross_encoder": 0.0,
            "llm_confidence": 0.0,
            "constraints": 0.0,
            "semantic_similarity": 0.0,
            "type_compatibility": 0.0,
            "unit_compatibility": 0.0
        }
        
        logger.info("Initialized hybrid scoring system")
    
    def compute_hybrid_score(self, 
                           source_field: SourceField,
                           candidate_doc: KnowledgeBaseDocument,
                           bi_encoder_score: float,
                           cross_encoder_score: float,
                           llm_confidence: float,
                           additional_features: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute hybrid score combining multiple signals.
        
        Args:
            source_field: Source field to match
            candidate_doc: Candidate document
            bi_encoder_score: Bi-encoder similarity score
            cross_encoder_score: Cross-encoder reranking score
            llm_confidence: LLM confidence score
            additional_features: Additional feature scores
            
        Returns:
            Tuple of (final_score, feature_scores)
        """
        additional_features = additional_features or {}
        
        # Compute individual feature scores
        feature_scores = {
            "bi_encoder": bi_encoder_score,
            "cross_encoder": cross_encoder_score,
            "llm_confidence": llm_confidence,
            "constraints": self._compute_constraints_score(source_field, candidate_doc),
            "semantic_similarity": self._compute_semantic_similarity_score(source_field, candidate_doc),
            "type_compatibility": self._compute_type_compatibility_score(source_field, candidate_doc),
            "unit_compatibility": self._compute_unit_compatibility_score(source_field, candidate_doc)
        }
        
        # Add additional features
        feature_scores.update(additional_features)
        
        # Compute weighted score
        final_score = (
            self._weights.bi_encoder * feature_scores["bi_encoder"] +
            self._weights.cross_encoder * feature_scores["cross_encoder"] +
            self._weights.llm_confidence * feature_scores["llm_confidence"] +
            self._weights.constraints * feature_scores["constraints"] +
            0.05 * feature_scores["semantic_similarity"] +  # Additional weight
            0.03 * feature_scores["type_compatibility"] +
            0.02 * feature_scores["unit_compatibility"]
        )
        
        # Apply calibration if available
        if self._is_calibrated and self._calibrator is not None:
            try:
                # Prepare features for calibration
                features_array = np.array([
                    feature_scores["bi_encoder"],
                    feature_scores["cross_encoder"],
                    feature_scores["llm_confidence"],
                    feature_scores["constraints"],
                    feature_scores["semantic_similarity"],
                    feature_scores["type_compatibility"],
                    feature_scores["unit_compatibility"]
                ]).reshape(1, -1)
                
                calibrated_score = self._calibrator.predict_proba(features_array)[0][1]
                final_score = calibrated_score
            except Exception as e:
                logger.warning(f"Calibration failed, using raw score: {e}")
        
        # Ensure score is in [0, 1] range
        final_score = max(0.0, min(1.0, final_score))
        
        logger.debug(f"Computed hybrid score: {final_score:.3f} for {candidate_doc.id}")
        return final_score, feature_scores
    
    def make_decision(self, 
                     source_field: SourceField,
                     candidates: List[CandidateMatch],
                     feature_scores_list: List[Dict[str, float]]) -> MatchingDecision:
        """
        Make final decision based on hybrid scoring.
        
        Args:
            source_field: Source field
            candidates: List of candidate matches
            feature_scores_list: List of feature scores for each candidate
            
        Returns:
            Final matching decision
        """
        if not candidates:
            return MatchingDecision(
                action=DecisionAction.REJECT,
                selected_target=None,
                final_confidence=0.0,
                guardrails=["no_candidates"]
            )
        
        # Find best candidate
        best_idx = 0
        best_score = 0.0
        
        for i, (candidate, feature_scores) in enumerate(zip(candidates, feature_scores_list)):
            if candidate.confidence_model > best_score:
                best_score = candidate.confidence_model
                best_idx = i
        
        best_candidate = candidates[best_idx]
        best_features = feature_scores_list[best_idx]
        
        # Apply decision thresholds
        action = self._apply_thresholds(best_score, best_candidate, source_field)
        
        # Build guardrails
        guardrails = self._build_guardrails(best_features, source_field, best_candidate)
        
        # Build review checklist if needed
        review_checklist = None
        if action == DecisionAction.REVIEW:
            review_checklist = self._build_review_checklist(source_field, best_candidate, best_features)
        
        return MatchingDecision(
            action=action,
            selected_target=best_candidate.target if action == DecisionAction.ACCEPT else None,
            final_confidence=best_score,
            guardrails=guardrails,
            review_checklist=review_checklist
        )
    
    def _compute_constraints_score(self, source_field: SourceField, 
                                 candidate_doc: KnowledgeBaseDocument) -> float:
        """Compute constraints validation score."""
        score = 0.5  # Base score
        
        constraints = candidate_doc.metadata.get("constraints", {})
        
        # Primary key constraint
        if constraints.get("is_primary_key"):
            if "id" in source_field.name_tokens or "key" in source_field.coarse_semantics:
                score += 0.3
            else:
                score -= 0.2
        
        # Foreign key constraint
        if constraints.get("is_foreign_key"):
            if self._has_corresponding_primary_key(source_field, candidate_doc):
                score += 0.2
            else:
                score -= 0.3
        
        # Not null constraint
        if constraints.get("not_null"):
            if source_field.inferred_type != "unknown":
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _compute_semantic_similarity_score(self, source_field: SourceField, 
                                        candidate_doc: KnowledgeBaseDocument) -> float:
        """Compute semantic similarity score."""
        score = 0.0
        
        # Name token similarity
        source_tokens = set(token.lower() for token in source_field.name_tokens)
        candidate_tokens = set(candidate_doc.column.lower().split('_'))
        
        if source_tokens & candidate_tokens:  # Intersection
            score += 0.4
        
        # Synonym matching
        synonyms = candidate_doc.metadata.get("synonyms", [])
        synonym_tokens = set()
        for synonym in synonyms:
            synonym_tokens.update(synonym.lower().split())
        
        if source_tokens & synonym_tokens:
            score += 0.3
        
        # Semantic hints matching
        hints = set(hint.lower() for hint in source_field.hints)
        description = candidate_doc.metadata.get("description", "").lower()
        
        for hint in hints:
            if hint in description:
                score += 0.2
                break
        
        # Coarse semantics matching
        coarse_semantics = set(sem.lower() for sem in source_field.coarse_semantics)
        for sem in coarse_semantics:
            if sem in description or sem in candidate_doc.column.lower():
                score += 0.1
                break
        
        return min(1.0, score)
    
    def _compute_type_compatibility_score(self, source_field: SourceField, 
                                       candidate_doc: KnowledgeBaseDocument) -> float:
        """Compute type compatibility score."""
        source_type = source_field.inferred_type
        candidate_type = candidate_doc.metadata.get("data_type", "").lower()
        
        # Type mapping
        type_mappings = {
            "datetime": ["timestamp", "datetime", "date", "time"],
            "integer": ["int", "integer", "bigint", "smallint"],
            "float": ["float", "double", "numeric", "decimal"],
            "text": ["varchar", "text", "char", "string"],
            "boolean": ["boolean", "bool"],
            "code": ["varchar", "char"],  # Codes are often strings
            "id": ["int", "integer", "bigint", "varchar"]
        }
        
        if source_type in type_mappings:
            compatible_types = type_mappings[source_type]
            if any(comp_type in candidate_type for comp_type in compatible_types):
                return 1.0
            else:
                return 0.3  # Partial compatibility
        else:
            return 0.5  # Unknown type, neutral score
    
    def _compute_unit_compatibility_score(self, source_field: SourceField, 
                                       candidate_doc: KnowledgeBaseDocument) -> float:
        """Compute unit compatibility score."""
        source_units = source_field.units
        candidate_units = candidate_doc.metadata.get("units")
        
        if not source_units or not candidate_units:
            return 0.5  # Neutral if no units specified
        
        # Normalize units
        source_norm = source_units.lower().strip()
        candidate_norm = candidate_units.lower().strip()
        
        # Exact match
        if source_norm == candidate_norm:
            return 1.0
        
        # Unit mappings
        unit_mappings = {
            "bpm": ["beats per minute", "beats/min", "hr"],
            "mmHg": ["mm Hg", "mmhg", "torr"],
            "°c": ["celsius", "c", "deg c"],
            "°f": ["fahrenheit", "f", "deg f"],
            "%": ["percent", "pct"],
            "/min": ["per minute", "per min", "/minute"]
        }
        
        for canonical, variants in unit_mappings.items():
            if (source_norm in variants or source_norm == canonical) and \
               (candidate_norm in variants or candidate_norm == canonical):
                return 1.0
        
        return 0.2  # Low compatibility for different units
    
    def _has_corresponding_primary_key(self, source_field: SourceField, 
                                    candidate_doc: KnowledgeBaseDocument) -> bool:
        """Check if source has corresponding primary key for foreign key."""
        fk_column = candidate_doc.column.lower()
        
        if "subject_id" in fk_column or "patient_id" in fk_column:
            return any("patient" in token.lower() or "subject" in token.lower() 
                      for token in source_field.neighbors)
        elif "hadm_id" in fk_column or "admission_id" in fk_column:
            return any("admission" in token.lower() or "encounter" in token.lower() 
                      for token in source_field.neighbors)
        elif "icustay_id" in fk_column:
            return any("icu" in token.lower() or "stay" in token.lower() 
                      for token in source_field.neighbors)
        
        return True  # Default to allowing if unsure
    
    def _apply_thresholds(self, score: float, candidate: CandidateMatch, 
                        source_field: SourceField) -> DecisionAction:
        """Apply decision thresholds."""
        # Basic threshold logic
        if score >= self._thresholds.accept_high:
            return DecisionAction.ACCEPT
        elif score >= self._thresholds.review_low:
            return DecisionAction.REVIEW
        else:
            return DecisionAction.REJECT
    
    def _build_guardrails(self, feature_scores: Dict[str, float], 
                         source_field: SourceField,
                         candidate: CandidateMatch) -> List[str]:
        """Build guardrails based on feature scores."""
        guardrails = []
        
        # Type compatibility guardrail
        if feature_scores["type_compatibility"] < 0.5:
            guardrails.append("type_incompatibility")
        
        # Unit compatibility guardrail
        if feature_scores["unit_compatibility"] < 0.5 and source_field.units:
            guardrails.append("unit_incompatibility")
        
        # Constraints guardrail
        if feature_scores["constraints"] < 0.5:
            guardrails.append("constraint_violation")
        
        # Semantic similarity guardrail
        if feature_scores["semantic_similarity"] < 0.3:
            guardrails.append("low_semantic_similarity")
        
        # LLM confidence guardrail
        if feature_scores["llm_confidence"] < 0.6:
            guardrails.append("low_llm_confidence")
        
        return guardrails
    
    def _build_review_checklist(self, source_field: SourceField, 
                              candidate: CandidateMatch,
                              feature_scores: Dict[str, float]) -> List[str]:
        """Build review checklist for human annotators."""
        checklist = [
            f"Vérifier la correspondance sémantique pour: {source_field.path}",
            f"Candidat sélectionné: {candidate.target}",
            f"Confiance finale: {candidate.confidence_model:.3f}"
        ]
        
        # Add specific items based on low scores
        if feature_scores["type_compatibility"] < 0.7:
            checklist.append("Vérifier la compatibilité des types de données")
        
        if feature_scores["unit_compatibility"] < 0.7 and source_field.units:
            checklist.append(f"Valider les unités: {source_field.units}")
        
        if feature_scores["semantic_similarity"] < 0.5:
            checklist.append("Vérifier la similarité sémantique")
        
        if feature_scores["constraints"] < 0.6:
            checklist.append("Vérifier les contraintes de base de données")
        
        return checklist
    
    def calibrate(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Calibrate the scoring system using training data.
        
        Args:
            training_data: List of training examples with features and labels
        """
        if not training_data:
            logger.warning("No training data provided for calibration")
            return
        
        try:
            # Extract features and labels
            X = []
            y = []
            
            for example in training_data:
                features = example["features"]
                label = 1 if example["label"] == "ACCEPT" else 0
                
                feature_vector = [
                    features.get("bi_encoder", 0.0),
                    features.get("cross_encoder", 0.0),
                    features.get("llm_confidence", 0.0),
                    features.get("constraints", 0.0),
                    features.get("semantic_similarity", 0.0),
                    features.get("type_compatibility", 0.0),
                    features.get("unit_compatibility", 0.0)
                ]
                
                X.append(feature_vector)
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Train calibrator
            base_classifier = LogisticRegression(random_state=42)
            self._calibrator = CalibratedClassifierCV(
                base_classifier, 
                method='isotonic',
                cv=3
            )
            self._calibrator.fit(X, y)
            
            self._is_calibrated = True
            logger.info(f"Calibration completed with {len(training_data)} examples")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self._is_calibrated = False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self._feature_importance.copy()
    
    def update_weights(self, new_weights: ScoringWeights) -> None:
        """Update scoring weights."""
        self._weights = new_weights
        logger.info("Updated scoring weights")
    
    def update_thresholds(self, new_thresholds: ScoringThresholds) -> None:
        """Update decision thresholds."""
        self._thresholds = new_thresholds
        logger.info("Updated decision thresholds")

