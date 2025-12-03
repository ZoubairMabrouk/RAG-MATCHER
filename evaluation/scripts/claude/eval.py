#!/usr/bin/env python3
"""
Evaluation script for RAG Virtual Rename against OMAP-MIMIC ground truth

Integrates with run_rag_virtual_rename.py output to compute:
- Accuracy@K metrics (standard for schema matching)
- Confusion matrix and classification metrics
- Detailed error analysis

Usage:
------
# Option 1: Evaluate from JSON output file
python scripts/evaluate_rag_matcher.py \
  --predictions artifacts/rag_mapping.json \
  --ground-truth datasets/omap/omop_mimic_data.xlsx \
  --level table \
  --threshold 0.5

# Option 2: Evaluate from log file
python scripts/evaluate_rag_matcher.py \
  --log-file logs/rag_matching.log \
  --ground-truth datasets/omap/omop_mimic_ground_truth.json \
  --level table

# Option 3: Run matching and evaluate in one go
python scripts/run_rag_virtual_rename.py \
  --uschema-file data/omop_uschema.json \
  --db-url postgresql://user:pass@localhost:5432/mimic \
  --out artifacts/rag_mapping.json && \
python scripts/evaluate_rag_matcher.py \
  --predictions artifacts/rag_mapping.json \
  --ground-truth datasets/omap/omop_mimic_data.xlsx
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    accuracy_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("evaluate_rag_matcher")


class OMAPGroundTruthLoader:
    """Load and normalize OMAP-MIMIC ground truth mappings"""
    
    @staticmethod
    def load_from_excel(file_path: str) -> Dict[str, str]:
        """
        Load ground truth from Excel file (omop_mimic_data.xlsx format)
        
        Expected columns: OMOP Table | MIMIC Table or similar
        """
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Try to identify source and target columns
            columns = df.columns.tolist()
            log.info(f"Excel columns: {columns}")
            
            # Common patterns for column names
            source_patterns = ['omop', 'source', 'from']
            target_patterns = ['mimic', 'target', 'to']
            
            source_col = None
            target_col = None
            
            for col in columns:
                col_lower = str(col).lower()
                if any(p in col_lower for p in source_patterns) and not source_col:
                    source_col = col
                elif any(p in col_lower for p in target_patterns) and not target_col:
                    target_col = col
            
            # Fallback: use first two columns
            if not source_col or not target_col:
                log.warning("Could not identify columns by name, using first two columns")
                source_col = columns[0]
                target_col = columns[1]
            
            log.info(f"Using columns: {source_col} -> {target_col}")
            
            ground_truth = {}
            for _, row in df.iterrows():
                source = str(row[source_col]).strip().lower()
                target = str(row[target_col]).strip().lower()
                
                # Skip invalid rows
                if source in ['nan', 'none', ''] or target in ['nan', 'none', '']:
                    continue
                    
                ground_truth[source] = target
            
            log.info(f"Loaded {len(ground_truth)} ground truth mappings from Excel")
            return ground_truth
            
        except Exception as e:
            log.error(f"Failed to load Excel file: {e}")
            raise
    
    @staticmethod
    def load_from_json(file_path: str) -> Dict[str, str]:
        """Load ground truth from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, dict):
            # Format 1: {"source1": "target1", "source2": "target2", ...}
            if all(isinstance(v, str) for v in data.values()):
                return {k.lower(): v.lower() for k, v in data.items()}
            # Format 2: {"mappings": [{...}]}
            elif "mappings" in data:
                gt = {}
                for m in data["mappings"]:
                    gt[m["source"].lower()] = m["target"].lower()
                return gt
        
        log.warning("Unknown JSON format, returning empty ground truth")
        return {}
    
    @staticmethod
    def load(file_path: str) -> Dict[str, str]:
        """Auto-detect format and load ground truth"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")
        
        if path.suffix in ['.xlsx', '.xls']:
            return OMAPGroundTruthLoader.load_from_excel(file_path)
        elif path.suffix == '.json':
            return OMAPGroundTruthLoader.load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class RAGMatcherOutputParser:
    """Parse different output formats from RAG matcher"""
    
    @staticmethod
    def parse_json_output(file_path: str, level: str = "table") -> List[Dict]:
        """
        Parse JSON output from run_rag_virtual_rename.py
        
        Args:
            file_path: Path to JSON output file
            level: "table" or "column" - which level to extract
        
        Returns:
            List of prediction dicts
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        predictions = []
        
        # Extract from the mapping section
        mappings = data.get("mapping", [])
        
        for entity_map in mappings:
            source = entity_map.get("entity", "").lower()
            target = entity_map.get("matched_table", "").lower()
            confidence = entity_map.get("table_confidence", 0.0)
            
            if level == "table" and source and target:
                predictions.append({
                    'source': source,
                    'target': target,
                    'confidence': confidence,
                    'rationale': entity_map.get("table_rationale", "")
                })
            
            # Also parse column-level if requested
            elif level == "column":
                table_name = target if target else source
                for attr in entity_map.get("attributes", []):
                    attr_name = attr.get("name", "").lower()
                    col_name = attr.get("target_column", "").lower()
                    col_conf = attr.get("confidence", 0.0)
                    
                    if attr_name and col_name:
                        predictions.append({
                            'source': f"{source}.{attr_name}",
                            'target': f"{table_name}.{col_name}",
                            'confidence': col_conf,
                            'rationale': attr.get("rationale", "")
                        })
        
        log.info(f"Parsed {len(predictions)} {level}-level predictions from JSON")
        return predictions
    
    @staticmethod
    def parse_log_file(file_path: str) -> List[Dict]:
        """
        Parse log file output
        
        Expected format:
        "Table match: source_table -> target_table (conf: 0.XXX)"
        """
        predictions = []
        
        with open(file_path, 'r') as f:
            for line in f:
                # Match pattern: "Table match: X -> Y (conf: Z)"
                if 'Table match:' in line or 'RAGSchemaMatcher' in line:
                    match = re.search(
                        r'Table match:\s*(\w+)\s*->\s*(\w+)\s*\(conf:\s*([\d.]+)\)',
                        line
                    )
                    if match:
                        source, target, conf = match.groups()
                        predictions.append({
                            'source': source.lower(),
                            'target': target.lower(),
                            'confidence': float(conf)
                        })
        
        log.info(f"Parsed {len(predictions)} predictions from log file")
        return predictions


class SchemaMatchingEvaluator:
    """
    Comprehensive evaluator for schema matching with OMAP dataset
    """
    
    def __init__(self, ground_truth: Dict[str, str]):
        self.ground_truth = ground_truth
        self.all_sources = set(ground_truth.keys())
        self.all_targets = set(ground_truth.values())
    
    def calculate_accuracy_at_k(
        self, 
        predictions: List[Dict], 
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Calculate Accuracy@K (Hit Rate@K) - standard metric for schema matching
        
        This measures: "Is the correct match in the top-K predictions?"
        """
        # Group predictions by source
        source_predictions = defaultdict(list)
        for pred in predictions:
            source_predictions[pred['source']].append(pred)
        
        # Sort each source's predictions by confidence (descending)
        for source in source_predictions:
            source_predictions[source].sort(
                key=lambda x: x['confidence'], 
                reverse=True
            )
        
        results = {}
        for k in k_values:
            hits = 0
            total = 0
            
            for source, gt_target in self.ground_truth.items():
                total += 1
                
                if source in source_predictions:
                    # Get top-K targets for this source
                    top_k_targets = [
                        p['target'] for p in source_predictions[source][:k]
                    ]
                    
                    # Check if ground truth is in top-K
                    if gt_target in top_k_targets:
                        hits += 1
            
            accuracy = hits / total if total > 0 else 0.0
            results[f'Accuracy@{k}'] = accuracy
        
        return results
    
    def calculate_mrr(self, predictions: List[Dict]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        MRR = (1/N) * Σ(1/rank_of_correct_answer)
        """
        source_predictions = defaultdict(list)
        for pred in predictions:
            source_predictions[pred['source']].append(pred)
        
        for source in source_predictions:
            source_predictions[source].sort(
                key=lambda x: x['confidence'], 
                reverse=True
            )
        
        reciprocal_ranks = []
        
        for source, gt_target in self.ground_truth.items():
            if source in source_predictions:
                # Find rank of correct target (1-indexed)
                targets = [p['target'] for p in source_predictions[source]]
                try:
                    rank = targets.index(gt_target) + 1
                    reciprocal_ranks.append(1.0 / rank)
                except ValueError:
                    # Correct target not in predictions
                    reciprocal_ranks.append(0.0)
            else:
                # No predictions for this source
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        return mrr
    
    def calculate_confusion_matrix_metrics(
        self, 
        predictions: List[Dict], 
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate confusion matrix and classification metrics
        
        Treats schema matching as binary classification:
        - Positive class: correct match
        - Negative class: incorrect match
        """
        # Build prediction lookup
        pred_dict = defaultdict(set)
        for pred in predictions:
            if pred['confidence'] >= confidence_threshold:
                pred_dict[pred['source']].add(pred['target'])
        
        y_true = []
        y_pred = []
        details = []
        
        # For each ground truth mapping
        for source, gt_target in self.ground_truth.items():
            predicted_targets = pred_dict.get(source, set())
            
            if gt_target in predicted_targets:
                # True Positive: correct match predicted
                y_true.append(1)
                y_pred.append(1)
                details.append({
                    'type': 'TP',
                    'source': source,
                    'predicted': gt_target,
                    'ground_truth': gt_target
                })
            else:
                # False Negative: missed correct match
                y_true.append(1)
                y_pred.append(0)
                details.append({
                    'type': 'FN',
                    'source': source,
                    'predicted': list(predicted_targets)[0] if predicted_targets else None,
                    'ground_truth': gt_target
                })
            
            # False Positives: incorrect matches predicted
            for pred_target in predicted_targets:
                if pred_target != gt_target:
                    y_true.append(0)
                    y_pred.append(1)
                    details.append({
                        'type': 'FP',
                        'source': source,
                        'predicted': pred_target,
                        'ground_truth': gt_target
                    })
        
        # Calculate metrics
        if not y_true:
            return {
                'confusion_matrix': np.array([[0, 0], [0, 0]]),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'details': []
            }
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        return {
            'confusion_matrix': cm,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            'details': details
        }
    
    def analyze_errors(
        self, 
        predictions: List[Dict], 
        confidence_threshold: float = 0.5
    ) -> Dict[str, List]:
        """Detailed error analysis"""
        pred_dict = defaultdict(list)
        for pred in predictions:
            if pred['confidence'] >= confidence_threshold:
                pred_dict[pred['source']].append(pred)
        
        errors = {
            'false_negatives': [],  # Missed matches
            'false_positives': [],  # Wrong matches
            'low_confidence_correct': [],  # Correct but below threshold
            'high_confidence_wrong': []  # Wrong but high confidence
        }
        
        for source, gt_target in self.ground_truth.items():
            if source not in pred_dict or not pred_dict[source]:
                # No predictions - False Negative
                errors['false_negatives'].append({
                    'source': source,
                    'expected': gt_target,
                    'predicted': None,
                    'reason': 'No predictions above threshold'
                })
            else:
                predictions_for_source = pred_dict[source]
                best_pred = max(predictions_for_source, key=lambda x: x['confidence'])
                
                if best_pred['target'] != gt_target:
                    # Wrong prediction
                    error_info = {
                        'source': source,
                        'expected': gt_target,
                        'predicted': best_pred['target'],
                        'confidence': best_pred['confidence']
                    }
                    
                    if best_pred['confidence'] >= 0.7:
                        errors['high_confidence_wrong'].append(error_info)
                    else:
                        errors['false_positives'].append(error_info)
        
        # Find correct matches that were below threshold
        all_predictions = defaultdict(list)
        for pred in predictions:  # Include all predictions
            all_predictions[pred['source']].append(pred)
        
        for source, gt_target in self.ground_truth.items():
            if source in all_predictions:
                for pred in all_predictions[source]:
                    if (pred['target'] == gt_target and 
                        pred['confidence'] < confidence_threshold):
                        errors['low_confidence_correct'].append({
                            'source': source,
                            'target': gt_target,
                            'confidence': pred['confidence']
                        })
        
        return errors
    
    def evaluate(
        self, 
        predictions: List[Dict],
        confidence_threshold: float = 0.5,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Run complete evaluation"""
        
        results = {
            'dataset_info': {
                'num_ground_truth': len(self.ground_truth),
                'num_predictions': len(predictions),
                'unique_sources_in_gt': len(self.all_sources),
                'unique_targets_in_gt': len(self.all_targets),
                'unique_sources_predicted': len(set(p['source'] for p in predictions)),
                'unique_targets_predicted': len(set(p['target'] for p in predictions))
            },
            'ranking_metrics': {},
            'classification_metrics': {},
            'error_analysis': {}
        }
        
        # Ranking metrics (primary for schema matching)
        results['ranking_metrics'] = self.calculate_accuracy_at_k(predictions, k_values)
        results['ranking_metrics']['MRR'] = self.calculate_mrr(predictions)
        
        # Classification metrics
        results['classification_metrics'] = self.calculate_confusion_matrix_metrics(
            predictions, confidence_threshold
        )
        
        # Error analysis
        results['error_analysis'] = self.analyze_errors(predictions, confidence_threshold)
        
        return results
    
    def print_report(self, results: Dict[str, Any], verbose: bool = False):
        """Print formatted evaluation report"""
        print("\n" + "=" * 80)
        print("OMAP-MIMIC SCHEMA MATCHING EVALUATION REPORT")
        print("=" * 80)
        
        # Dataset info
        info = results['dataset_info']
        print(f"\n{'Dataset Statistics':^80}")
        print("-" * 80)
        print(f"  Ground Truth Mappings: {info['num_ground_truth']}")
        print(f"  Total Predictions: {info['num_predictions']}")
        print(f"  Sources in GT: {info['unique_sources_in_gt']}")
        print(f"  Targets in GT: {info['unique_targets_in_gt']}")
        
        # Ranking metrics
        print(f"\n{'RANKING METRICS (Primary for Schema Matching)':^80}")
        print("-" * 80)
        for metric, value in results['ranking_metrics'].items():
            percentage = f"({value*100:.2f}%)"
            print(f"  {metric:<20}: {value:.4f} {percentage:>10}")
        
        # Classification metrics
        print(f"\n{'CLASSIFICATION METRICS':^80}")
        print("-" * 80)
        cm = results['classification_metrics']
        print(f"  Precision: {cm['precision']:.4f}")
        print(f"  Recall:    {cm['recall']:.4f}")
        print(f"  F1-Score:  {cm['f1_score']:.4f}")
        print(f"  Accuracy:  {cm['accuracy']:.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"                      Predicted Negative    Predicted Positive")
        print(f"  Actual Negative:    {cm['true_negatives']:>8}             {cm['false_positives']:>8}")
        print(f"  Actual Positive:    {cm['false_negatives']:>8}             {cm['true_positives']:>8}")
        
        # Error analysis summary
        print(f"\n{'ERROR ANALYSIS':^80}")
        print("-" * 80)
        errors = results['error_analysis']
        print(f"  False Negatives (Missed):          {len(errors['false_negatives'])}")
        print(f"  False Positives (Wrong):           {len(errors['false_positives'])}")
        print(f"  High Confidence Wrong:             {len(errors['high_confidence_wrong'])}")
        print(f"  Low Confidence Correct:            {len(errors['low_confidence_correct'])}")
        
        if verbose:
            self._print_detailed_errors(errors)
        
        print("\n" + "=" * 80)
    
    def _print_detailed_errors(self, errors: Dict):
        """Print detailed error examples"""
        print(f"\n{'Detailed Error Examples':^80}")
        print("-" * 80)
        
        if errors['false_negatives']:
            print("\nFalse Negatives (Top 5):")
            for err in errors['false_negatives'][:5]:
                print(f"  ✗ {err['source']} -> expected: {err['expected']}, got: {err['predicted']}")
        
        if errors['high_confidence_wrong']:
            print("\nHigh Confidence Wrong (Top 5):")
            for err in errors['high_confidence_wrong'][:5]:
                print(f"  ✗ {err['source']} -> expected: {err['expected']}, "
                      f"predicted: {err['predicted']} (conf: {err['confidence']:.3f})")
        
        if errors['low_confidence_correct']:
            print("\nLow Confidence Correct (consider lowering threshold):")
            for err in errors['low_confidence_correct'][:5]:
                print(f"  ⚠ {err['source']} -> {err['target']} (conf: {err['confidence']:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG Schema Matcher against OMAP-MIMIC ground truth"
    )
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--predictions',
        help='Path to JSON output from run_rag_virtual_rename.py'
    )
    input_group.add_argument(
        '--log-file',
        help='Path to log file with matching results'
    )
    
    # Ground truth
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to OMAP ground truth file (Excel or JSON)'
    )
    
    # Options
    parser.add_argument(
        '--level',
        choices=['table', 'column'],
        default='table',
        help='Evaluation level: table or column matching'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for binary classification'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='K values for Accuracy@K calculation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed error analysis'
    )
    parser.add_argument(
        '--output',
        help='Save evaluation results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load ground truth
    log.info(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = OMAPGroundTruthLoader.load(args.ground_truth)
    log.info(f"Loaded {len(ground_truth)} ground truth mappings")
    
    # Parse predictions
    if args.predictions:
        log.info(f"Parsing predictions from JSON: {args.predictions}")
        predictions = RAGMatcherOutputParser.parse_json_output(
            args.predictions, 
            level=args.level
        )
    else:
        log.info(f"Parsing predictions from log: {args.log_file}")
        predictions = RAGMatcherOutputParser.parse_log_file(args.log_file)
    
    if not predictions:
        log.error("No predictions found!")
        return 1
    
    # Run evaluation
    evaluator = SchemaMatchingEvaluator(ground_truth)
    results = evaluator.evaluate(
        predictions,
        confidence_threshold=args.threshold,
        k_values=args.k_values
    )
    
    # Print report
    evaluator.print_report(results, verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = json.loads(
            json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        )
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        log.info(f"\nSaved detailed results to: {output_path}")
    
    # Return exit code based on performance
    # Consider it a failure if Accuracy@1 < 0.5
    accuracy_at_1 = results['ranking_metrics'].get('Accuracy@1', 0)
    return 0 if accuracy_at_1 >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())