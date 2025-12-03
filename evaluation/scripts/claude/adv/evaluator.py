#!/usr/bin/env python3
"""
Column-level evaluation for OMAP-MIMIC dataset

The OMAP dataset format:
- omop: OMOP table-column pair (e.g., "person-person_id")  
- table: MIMIC table-column pair (e.g., "admissions-subject_id")
- label: 1 if they match, 0 if they don't

Usage:
------
python scripts/evaluate_omap_column_matching.py \
    --ground-truth omop_mimic_data.xlsx \
    --predictions artifacts/rag_mapping.json \
    --threshold 0.5 \
    --output results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("omap_column_evaluator")


class OMAPColumnEvaluator:
    """Evaluator for column-level schema matching against OMAP-MIMIC"""
    
    def __init__(self, ground_truth_file: str):
        """
        Load OMAP-MIMIC ground truth
        
        Args:
            ground_truth_file: Path to omop_mimic_data.xlsx
        """
        self.df = pd.read_excel(ground_truth_file)
        log.info(f"Loaded ground truth: {len(self.df)} pairs, {self.df['label'].sum()} positive matches")
        
        # Create lookup structures for efficient evaluation
        self.positive_matches = self._build_positive_matches()
        self.all_pairs = self._build_all_pairs()
        
    def _build_positive_matches(self) -> Dict[str, Set[str]]:
        """
        Build mapping of OMOP table-column -> set of matching MIMIC table-columns
        
        Returns:
            Dict like {"person-person_id": {"patients-subject_id", ...}}
        """
        positive = defaultdict(set)
        
        for _, row in self.df[self.df['label'] == 1].iterrows():
            omop_pair = str(row['omop']).strip().lower()
            mimic_pair = str(row['table']).strip().lower()
            positive[omop_pair].add(mimic_pair)
        
        log.info(f"Found {len(positive)} OMOP columns with matches")
        return dict(positive)
    
    def _build_all_pairs(self) -> Set[Tuple[str, str]]:
        """Build set of all (omop, mimic) pairs in dataset"""
        pairs = set()
        for _, row in self.df.iterrows():
            omop = str(row['omop']).strip().lower()
            mimic = str(row['table']).strip().lower()
            pairs.add((omop, mimic))
        return pairs
    
    def parse_rag_predictions(self, predictions_file: str) -> List[Dict]:
        """
        Parse RAG matcher output into column-level predictions
        
        Expected format from run_rag_virtual_rename.py:
        {
            "mapping": [
                {
                    "entity": "person",
                    "matched_table": "patients",
                    "table_confidence": 0.85,
                    "attributes": [
                        {
                            "name": "person_id",
                            "target_column": "subject_id",
                            "confidence": 0.92
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        
        Returns:
            List of predictions: [
                {
                    "omop_table": "person",
                    "omop_column": "person_id",
                    "mimic_table": "patients",
                    "mimic_column": "subject_id",
                    "confidence": 0.92,
                    "omop_pair": "person-person_id",
                    "mimic_pair": "patients-subject_id"
                },
                ...
            ]
        """
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        predictions = []
        
        for entity_map in data.get("mapping", []):
            omop_table = entity_map.get("entity", "").lower()
            mimic_table = entity_map.get("matched_table", "").lower()
            
            if not mimic_table:
                continue  # No table match, skip
            
            for attr in entity_map.get("attributes", []):
                omop_column = attr.get("name", "").lower()
                mimic_column = attr.get("target_column", "").lower()
                confidence = attr.get("confidence", 0.0)
                
                if mimic_column:  # Only include if column was matched
                    predictions.append({
                        "omop_table": omop_table,
                        "omop_column": omop_column,
                        "mimic_table": mimic_table,
                        "mimic_column": mimic_column,
                        "confidence": confidence,
                        "omop_pair": f"{omop_table}-{omop_column}",
                        "mimic_pair": f"{mimic_table}-{mimic_column}"
                    })
        
        log.info(f"Parsed {len(predictions)} column-level predictions")
        return predictions
    
    def evaluate_at_threshold(
        self, 
        predictions: List[Dict], 
        threshold: float
    ) -> Dict:
        """
        Evaluate predictions at a specific confidence threshold
        
        Args:
            predictions: List of prediction dicts
            threshold: Confidence threshold for positive prediction
            
        Returns:
            Dict with evaluation metrics
        """
        # Build prediction lookup
        predicted_matches = defaultdict(set)
        for pred in predictions:
            if pred['confidence'] >= threshold:
                omop_pair = pred['omop_pair']
                mimic_pair = pred['mimic_pair']
                predicted_matches[omop_pair].add(mimic_pair)
        
        # Calculate metrics
        y_true = []
        y_pred = []
        y_scores = []  # For AUC metrics
        
        tp_examples = []
        fp_examples = []
        fn_examples = []
        
        # For each pair in the dataset, determine TP/FP/FN/TN
        for omop_pair, true_matches in self.positive_matches.items():
            predicted = predicted_matches.get(omop_pair, set())
            
            # True Positives: predicted and actually match
            for mimic_pair in predicted & true_matches:
                y_true.append(1)
                y_pred.append(1)
                
                # Find confidence
                conf = next(
                    (p['confidence'] for p in predictions 
                     if p['omop_pair'] == omop_pair and p['mimic_pair'] == mimic_pair),
                    0.0
                )
                y_scores.append(conf)
                
                tp_examples.append({
                    'omop': omop_pair,
                    'mimic': mimic_pair,
                    'confidence': conf
                })
            
            # False Positives: predicted but don't actually match
            for mimic_pair in predicted - true_matches:
                y_true.append(0)
                y_pred.append(1)
                
                conf = next(
                    (p['confidence'] for p in predictions 
                     if p['omop_pair'] == omop_pair and p['mimic_pair'] == mimic_pair),
                    0.0
                )
                y_scores.append(conf)
                
                fp_examples.append({
                    'omop': omop_pair,
                    'mimic': mimic_pair,
                    'confidence': conf,
                    'expected': list(true_matches)
                })
            
            # False Negatives: should match but weren't predicted
            for mimic_pair in true_matches - predicted:
                y_true.append(1)
                y_pred.append(0)
                y_scores.append(0.0)  # Not predicted = 0 confidence
                
                fn_examples.append({
                    'omop': omop_pair,
                    'mimic': mimic_pair,
                    'reason': 'not predicted or below threshold'
                })
        
        # Calculate metrics
        if not y_true:
            return {'error': 'No ground truth matches found'}
        
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': cm.tolist(),
            'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            'total_predictions': len([p for p in predictions if p['confidence'] >= threshold]),
            'total_ground_truth': len(y_true),
            'positive_ground_truth': sum(y_true),
            'examples': {
                'true_positives': tp_examples[:5],  # Top 5 examples
                'false_positives': fp_examples[:5],
                'false_negatives': fn_examples[:5]
            }
        }
        
        # Calculate AUC metrics if we have scores
        if y_scores and len(set(y_true)) > 1:  # Need both classes
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['avg_precision'] = average_precision_score(y_true, y_scores)
            except:
                pass
        
        return metrics
    
    def evaluate_multiple_thresholds(
        self,
        predictions: List[Dict],
        thresholds: List[float] = None
    ) -> Dict:
        """
        Evaluate across multiple thresholds
        
        Args:
            predictions: List of prediction dicts
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9)
            
        Returns:
            Dict with results for each threshold
        """
        if thresholds is None:
            thresholds = [i * 0.1 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
        
        results = {}
        best_f1 = 0
        best_threshold = 0
        
        for threshold in thresholds:
            log.info(f"Evaluating at threshold: {threshold:.2f}")
            metrics = self.evaluate_at_threshold(predictions, threshold)
            results[threshold] = metrics
            
            if metrics.get('f1_score', 0) > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
        
        return {
            'per_threshold': results,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'summary': self._create_summary(results)
        }
    
    def _create_summary(self, results: Dict) -> pd.DataFrame:
        """Create summary DataFrame of results across thresholds"""
        summary_data = []
        
        for threshold, metrics in results.items():
            if 'error' not in metrics:
                summary_data.append({
                    'threshold': threshold,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'tp': metrics['true_positives'],
                    'fp': metrics['false_positives'],
                    'fn': metrics['false_negatives'],
                    'predictions': metrics['total_predictions']
                })
        
        return pd.DataFrame(summary_data)
    
    def print_report(self, results: Dict, verbose: bool = False):
        """Print evaluation report"""
        print("\n" + "=" * 80)
        print("OMAP-MIMIC COLUMN-LEVEL EVALUATION REPORT")
        print("=" * 80)
        
        if 'per_threshold' in results:
            # Multi-threshold evaluation
            print(f"\nBest Threshold: {results['best_threshold']:.2f}")
            print(f"Best F1-Score: {results['best_f1']:.4f}")
            
            print("\n" + "=" * 80)
            print("RESULTS ACROSS THRESHOLDS")
            print("=" * 80)
            
            summary_df = results['summary']
            print(summary_df.to_string(index=False))
            
            if verbose and 'per_threshold' in results:
                best_threshold = results['best_threshold']
                best_metrics = results['per_threshold'][best_threshold]
                
                print(f"\n" + "=" * 80)
                print(f"DETAILED RESULTS AT BEST THRESHOLD ({best_threshold:.2f})")
                print("=" * 80)
                
                self._print_threshold_details(best_metrics)
        
        else:
            # Single threshold evaluation
            self._print_threshold_details(results)
    
    def _print_threshold_details(self, metrics: Dict):
        """Print details for a single threshold"""
        print(f"\nMetrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"                    Predicted Negative    Predicted Positive")
        print(f"  Actual Negative:  {cm[0][0]:>8}             {cm[0][1]:>8}")
        print(f"  Actual Positive:  {cm[1][0]:>8}             {cm[1][1]:>8}")
        
        print(f"\nCounts:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        
        # Examples
        examples = metrics.get('examples', {})
        
        if examples.get('true_positives'):
            print(f"\nTrue Positive Examples (correct matches):")
            for ex in examples['true_positives'][:3]:
                print(f"  ✓ {ex['omop']} -> {ex['mimic']} (conf: {ex['confidence']:.3f})")
        
        if examples.get('false_positives'):
            print(f"\nFalse Positive Examples (incorrect matches):")
            for ex in examples['false_positives'][:3]:
                print(f"  ✗ {ex['omop']} -> {ex['mimic']} (conf: {ex['confidence']:.3f})")
                if ex.get('expected'):
                    print(f"    Expected: {ex['expected'][:2]}")
        
        if examples.get('false_negatives'):
            print(f"\nFalse Negative Examples (missed matches):")
            for ex in examples['false_negatives'][:3]:
                print(f"  ✗ {ex['omop']} -> {ex['mimic']} (not predicted)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG schema matcher against OMAP-MIMIC"
    )
    
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to omop_mimic_data.xlsx'
    )
    parser.add_argument(
        '--predictions',
        required=True,
        help='Path to RAG predictions JSON'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Single threshold to evaluate (optional)'
    )
    parser.add_argument(
        '--threshold-range',
        nargs=3,
        type=float,
        metavar=('START', 'END', 'STEP'),
        default=[0.1, 1.0, 0.1],
        help='Threshold range: start end step (default: 0.1 1.0 0.1)'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed results'
    )
    
    args = parser.parse_args()
    
    # Load evaluator
    evaluator = OMAPColumnEvaluator(args.ground_truth)
    
    # Parse predictions
    predictions = evaluator.parse_rag_predictions(args.predictions)
    
    # Evaluate
    if args.threshold is not None:
        # Single threshold
        log.info(f"Evaluating at threshold: {args.threshold}")
        results = evaluator.evaluate_at_threshold(predictions, args.threshold)
    else:
        # Multiple thresholds
        start, end, step = args.threshold_range
        thresholds = np.arange(start, end + step/2, step).tolist()
        log.info(f"Evaluating across thresholds: {thresholds}")
        results = evaluator.evaluate_multiple_thresholds(predictions, thresholds)
    
    # Print report
    evaluator.print_report(results, verbose=args.verbose)
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=convert_types)
        
        log.info(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())