"""
Evaluation metrics and datasets for RAG schema matching.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from src.domain.entities.rag_schema import (
    SchemaMatchingResult, DecisionAction, SourceField, FieldType
)

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Evaluation metrics for RAG schema matching system.
    Single Responsibility: Metrics calculation and evaluation.
    """
    
    def __init__(self):
        self.metrics = {
            "accuracy": {},
            "precision": {},
            "recall": {},
            "f1": {},
            "coverage": {},
            "efficiency": {},
            "calibration": {}
        }
    
    def evaluate_results(self, 
                        results: List[SchemaMatchingResult],
                        ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate matching results against ground truth.
        
        Args:
            results: List of matching results
            ground_truth: List of ground truth annotations
            
        Returns:
            Comprehensive evaluation metrics
        """
        logger.info(f"Evaluating {len(results)} results against ground truth")
        
        # Calculate different types of metrics
        accuracy_metrics = self._calculate_accuracy_metrics(results, ground_truth)
        precision_recall_metrics = self._calculate_precision_recall_metrics(results, ground_truth)
        coverage_metrics = self._calculate_coverage_metrics(results, ground_truth)
        efficiency_metrics = self._calculate_efficiency_metrics(results)
        calibration_metrics = self._calculate_calibration_metrics(results, ground_truth)
        
        # Combine all metrics
        evaluation_results = {
            "accuracy": accuracy_metrics,
            "precision_recall": precision_recall_metrics,
            "coverage": coverage_metrics,
            "efficiency": efficiency_metrics,
            "calibration": calibration_metrics,
            "summary": self._create_summary_metrics(
                accuracy_metrics, precision_recall_metrics, coverage_metrics
            )
        }
        
        logger.info("Evaluation completed")
        return evaluation_results
    
    def _calculate_accuracy_metrics(self, 
                                  results: List[SchemaMatchingResult],
                                  ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy metrics."""
        # Create ground truth lookup
        gt_lookup = {gt["source_field"]: gt for gt in ground_truth}
        
        correct_predictions = 0
        total_predictions = 0
        
        # Top-1 accuracy
        top1_correct = 0
        top1_total = 0
        
        # Top-k accuracy (k=5)
        topk_correct = 0
        topk_total = 0
        
        for result in results:
            if result.source_field not in gt_lookup:
                continue
            
            gt = gt_lookup[result.source_field]
            total_predictions += 1
            
            # Exact match accuracy
            if (result.decision.action == DecisionAction.ACCEPT and 
                result.decision.selected_target == gt.get("correct_target")):
                correct_predictions += 1
            
            # Top-1 accuracy (best candidate)
            top1_total += 1
            if result.candidates:
                best_candidate = result.candidates[0]
                if best_candidate.target == gt.get("correct_target"):
                    top1_correct += 1
            
            # Top-k accuracy
            topk_total += 1
            top_k_candidates = result.candidates[:5]  # Top 5
            if any(candidate.target == gt.get("correct_target") for candidate in top_k_candidates):
                topk_correct += 1
        
        return {
            "exact_match_accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0.0,
            "top1_accuracy": top1_correct / top1_total if top1_total > 0 else 0.0,
            "top5_accuracy": topk_correct / topk_total if topk_total > 0 else 0.0,
            "total_predictions": total_predictions
        }
    
    def _calculate_precision_recall_metrics(self, 
                                          results: List[SchemaMatchingResult],
                                          ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate precision and recall metrics."""
        # Create ground truth lookup
        gt_lookup = {gt["source_field"]: gt for gt in ground_truth}
        
        # Confusion matrix components
        true_positives = 0  # Correct ACCEPT decisions
        false_positives = 0  # Incorrect ACCEPT decisions
        false_negatives = 0  # Should have been ACCEPT but wasn't
        true_negatives = 0  # Correct REJECT decisions
        
        # Decision-specific metrics
        accept_decisions = 0
        review_decisions = 0
        reject_decisions = 0
        
        for result in results:
            if result.source_field not in gt_lookup:
                continue
            
            gt = gt_lookup[result.source_field]
            correct_target = gt.get("correct_target")
            should_accept = gt.get("should_accept", True)
            
            # Count decisions
            if result.decision.action == DecisionAction.ACCEPT:
                accept_decisions += 1
                if result.decision.selected_target == correct_target:
                    true_positives += 1
                else:
                    false_positives += 1
            elif result.decision.action == DecisionAction.REVIEW:
                review_decisions += 1
                if should_accept:
                    false_negatives += 1
            else:  # REJECT
                reject_decisions += 1
                if not should_accept:
                    true_negatives += 1
                else:
                    false_negatives += 1
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "decision_distribution": {
                "accept": accept_decisions,
                "review": review_decisions,
                "reject": reject_decisions
            }
        }
    
    def _calculate_coverage_metrics(self, 
                                  results: List[SchemaMatchingResult],
                                  ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate coverage metrics."""
        # Create ground truth lookup
        gt_lookup = {gt["source_field"]: gt for gt in ground_truth}
        
        # Coverage metrics
        total_fields = len(ground_truth)
        processed_fields = len([r for r in results if r.source_field in gt_lookup])
        
        # Decision coverage
        accept_coverage = len([r for r in results 
                             if r.source_field in gt_lookup and 
                             r.decision.action == DecisionAction.ACCEPT])
        
        # Field type coverage
        type_coverage = {}
        for result in results:
            if result.source_field not in gt_lookup:
                continue
            
            gt = gt_lookup[result.source_field]
            field_type = gt.get("field_type", "unknown")
            
            if field_type not in type_coverage:
                type_coverage[field_type] = {"total": 0, "matched": 0}
            
            type_coverage[field_type]["total"] += 1
            if result.decision.action in [DecisionAction.ACCEPT, DecisionAction.REVIEW]:
                type_coverage[field_type]["matched"] += 1
        
        # Calculate coverage rates
        for field_type in type_coverage:
            total = type_coverage[field_type]["total"]
            matched = type_coverage[field_type]["matched"]
            type_coverage[field_type]["coverage_rate"] = matched / total if total > 0 else 0.0
        
        return {
            "total_fields": total_fields,
            "processed_fields": processed_fields,
            "processing_coverage": processed_fields / total_fields if total_fields > 0 else 0.0,
            "accept_coverage": accept_coverage / total_fields if total_fields > 0 else 0.0,
            "type_coverage": type_coverage
        }
    
    def _calculate_efficiency_metrics(self, results: List[SchemaMatchingResult]) -> Dict[str, float]:
        """Calculate efficiency metrics."""
        if not results:
            return {}
        
        # Processing time metrics
        processing_times = [r.processing_time_ms for r in results if r.processing_time_ms > 0]
        
        # Candidate count metrics
        candidate_counts = [len(r.candidates) for r in results]
        
        # Confidence score metrics
        confidence_scores = []
        for result in results:
            if result.candidates:
                confidence_scores.extend([c.confidence_model for c in result.candidates])
        
        return {
            "avg_processing_time_ms": np.mean(processing_times) if processing_times else 0.0,
            "median_processing_time_ms": np.median(processing_times) if processing_times else 0.0,
            "max_processing_time_ms": np.max(processing_times) if processing_times else 0.0,
            "avg_candidates_per_field": np.mean(candidate_counts) if candidate_counts else 0.0,
            "avg_confidence_score": np.mean(confidence_scores) if confidence_scores else 0.0,
            "confidence_std": np.std(confidence_scores) if confidence_scores else 0.0
        }
    
    def _calculate_calibration_metrics(self, 
                                     results: List[SchemaMatchingResult],
                                     ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate calibration metrics."""
        # Create ground truth lookup
        gt_lookup = {gt["source_field"]: gt for gt in ground_truth}
        
        # Bin confidence scores and calculate accuracy per bin
        confidence_bins = np.arange(0.0, 1.1, 0.1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_low = confidence_bins[i]
            bin_high = confidence_bins[i + 1]
            
            bin_results = []
            bin_confidences_list = []
            
            for result in results:
                if result.source_field not in gt_lookup:
                    continue
                
                if result.candidates:
                    best_confidence = result.candidates[0].confidence_model
                    if bin_low <= best_confidence < bin_high:
                        gt = gt_lookup[result.source_field]
                        is_correct = (result.decision.action == DecisionAction.ACCEPT and 
                                    result.decision.selected_target == gt.get("correct_target"))
                        bin_results.append(is_correct)
                        bin_confidences_list.append(best_confidence)
            
            if bin_results:
                bin_accuracy = np.mean(bin_results)
                bin_confidence = np.mean(bin_confidences_list)
                bin_count = len(bin_results)
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)
        
        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = sum(bin_counts)
        
        for accuracy, confidence, count in zip(bin_accuracies, bin_confidences, bin_counts):
            ece += (count / total_samples) * abs(accuracy - confidence)
        
        return {
            "expected_calibration_error": ece,
            "confidence_bins": {
                "accuracies": bin_accuracies,
                "confidences": bin_confidences,
                "counts": bin_counts
            }
        }
    
    def _create_summary_metrics(self, 
                              accuracy_metrics: Dict[str, float],
                              precision_recall_metrics: Dict[str, float],
                              coverage_metrics: Dict[str, float]) -> Dict[str, float]:
        """Create summary metrics."""
        return {
            "overall_accuracy": accuracy_metrics.get("exact_match_accuracy", 0.0),
            "top5_accuracy": accuracy_metrics.get("top5_accuracy", 0.0),
            "precision": precision_recall_metrics.get("precision", 0.0),
            "recall": precision_recall_metrics.get("recall", 0.0),
            "f1_score": precision_recall_metrics.get("f1_score", 0.0),
            "coverage_rate": coverage_metrics.get("processing_coverage", 0.0),
            "accept_rate": precision_recall_metrics.get("decision_distribution", {}).get("accept", 0) / 
                          sum(precision_recall_metrics.get("decision_distribution", {}).values()) if 
                          sum(precision_recall_metrics.get("decision_distribution", {}).values()) > 0 else 0.0
        }


class EvaluationDataset:
    """
    Evaluation dataset for RAG schema matching.
    Single Responsibility: Dataset management and loading.
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.ground_truth = []
        self.test_cases = []
    
    def load_dataset(self, dataset_path: str) -> None:
        """Load evaluation dataset from file."""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.ground_truth = data.get("ground_truth", [])
            self.test_cases = data.get("test_cases", [])
            
            logger.info(f"Loaded dataset with {len(self.ground_truth)} ground truth entries")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def save_dataset(self, dataset_path: str) -> None:
        """Save evaluation dataset to file."""
        try:
            data = {
                "ground_truth": self.ground_truth,
                "test_cases": self.test_cases,
                "metadata": {
                    "total_entries": len(self.ground_truth),
                    "total_test_cases": len(self.test_cases)
                }
            }
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved dataset to {dataset_path}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def add_ground_truth(self, 
                        source_field: str,
                        correct_target: str,
                        field_type: str,
                        should_accept: bool = True,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add ground truth entry."""
        entry = {
            "source_field": source_field,
            "correct_target": correct_target,
            "field_type": field_type,
            "should_accept": should_accept,
            "metadata": metadata or {}
        }
        
        self.ground_truth.append(entry)
    
    def add_test_case(self, 
                     source_field: SourceField,
                     expected_target: Optional[str] = None) -> None:
        """Add test case."""
        test_case = {
            "source_field": source_field.dict(),
            "expected_target": expected_target
        }
        
        self.test_cases.append(test_case)
    
    def create_synthetic_dataset(self, num_cases: int = 100) -> None:
        """Create synthetic evaluation dataset."""
        logger.info(f"Creating synthetic dataset with {num_cases} cases")
        
        # Common field patterns for MIMIC-III
        field_patterns = [
            # Patient identifiers
            {"path": "patient.id", "name_tokens": ["patient", "id"], "type": "id", "target": "PATIENTS.SUBJECT_ID"},
            {"path": "patient.subject_id", "name_tokens": ["subject", "id"], "type": "id", "target": "PATIENTS.SUBJECT_ID"},
            
            # Admission information
            {"path": "admission.date", "name_tokens": ["admission", "date"], "type": "datetime", "target": "ADMISSIONS.ADMITTIME"},
            {"path": "admission.time", "name_tokens": ["admit", "time"], "type": "datetime", "target": "ADMISSIONS.ADMITTIME"},
            {"path": "admission.id", "name_tokens": ["admission", "id"], "type": "id", "target": "ADMISSIONS.HADM_ID"},
            
            # Vital signs
            {"path": "vitals.heart_rate", "name_tokens": ["heart", "rate"], "type": "integer", "target": "CHARTEVENTS.VALUENUM"},
            {"path": "vitals.bp_systolic", "name_tokens": ["bp", "systolic"], "type": "integer", "target": "CHARTEVENTS.VALUENUM"},
            {"path": "vitals.temperature", "name_tokens": ["temperature", "temp"], "type": "float", "target": "CHARTEVENTS.VALUENUM"},
            
            # Lab values
            {"path": "labs.hemoglobin", "name_tokens": ["hemoglobin", "hgb"], "type": "float", "target": "LABEVENTS.VALUENUM"},
            {"path": "labs.creatinine", "name_tokens": ["creatinine"], "type": "float", "target": "LABEVENTS.VALUENUM"},
            {"path": "labs.units", "name_tokens": ["units", "unit"], "type": "text", "target": "LABEVENTS.VALUEUOM"},
            
            # Diagnoses
            {"path": "diagnosis.code", "name_tokens": ["diagnosis", "code"], "type": "code", "target": "DIAGNOSES_ICD.ICD9_CODE"},
            {"path": "diagnosis.primary", "name_tokens": ["primary", "diagnosis"], "type": "text", "target": "DIAGNOSES_ICD.SHORT_TITLE"},
            
            # Medications
            {"path": "medication.drug", "name_tokens": ["drug", "medication"], "type": "text", "target": "PRESCRIPTIONS.DRUG"},
            {"path": "medication.ndc", "name_tokens": ["ndc", "code"], "type": "code", "target": "PRESCRIPTIONS.DRUG_TYPE"},
        ]
        
        # Generate synthetic cases
        for i in range(num_cases):
            pattern = field_patterns[i % len(field_patterns)]
            
            # Create source field
            source_field = SourceField(
                path=pattern["path"],
                name_tokens=pattern["name_tokens"],
                inferred_type=FieldType(pattern["type"]),
                hints=pattern["name_tokens"],
                coarse_semantics=[pattern["type"]]
            )
            
            # Add test case
            self.add_test_case(source_field, pattern["target"])
            
            # Add ground truth
            self.add_ground_truth(
                source_field=pattern["path"],
                correct_target=pattern["target"],
                field_type=pattern["type"],
                should_accept=True
            )
        
        logger.info(f"Created {num_cases} synthetic test cases")
    
    def get_ground_truth_for_field(self, source_field: str) -> Optional[Dict[str, Any]]:
        """Get ground truth for specific field."""
        for gt in self.ground_truth:
            if gt["source_field"] == source_field:
                return gt
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        field_types = {}
        for gt in self.ground_truth:
            field_type = gt.get("field_type", "unknown")
            field_types[field_type] = field_types.get(field_type, 0) + 1
        
        return {
            "total_ground_truth": len(self.ground_truth),
            "total_test_cases": len(self.test_cases),
            "field_type_distribution": field_types,
            "accept_rate": sum(1 for gt in self.ground_truth if gt.get("should_accept", True)) / len(self.ground_truth) if self.ground_truth else 0.0
        }

