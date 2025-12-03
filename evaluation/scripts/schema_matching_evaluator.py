import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import json
from collections import defaultdict

class SchemaMatchingEvaluator:
    """
    Evaluator for table-level schema matching based on OMAP-MIMIC dataset
    """
    
    def __init__(self, ground_truth_file):
        """
        Initialize evaluator with ground truth mappings
        
        Args:
            ground_truth_file: Path to omop_mimic_data.xlsx or parsed ground truth
        """
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        
    def load_ground_truth(self, file_path):
        """
        Load ground truth mappings from Excel file
        Expected format: source_table -> target_table mapping
        
        Returns:
            dict: {source_table: target_table}
        """
        try:
            # Try loading as Excel
            df = pd.read_excel(file_path)
            # Adjust column names based on actual file structure
            # Common formats: 'MIMIC Table', 'OMOP Table' or similar
            ground_truth = {}
            for _, row in df.iterrows():
                source = str(row.iloc[0]).strip().lower()  # First column
                target = str(row.iloc[1]).strip().lower()  # Second column
                if source != 'nan' and target != 'nan':
                    ground_truth[source] = target
            return ground_truth
        except:
            # If Excel fails, try loading as dict/JSON
            if isinstance(file_path, dict):
                return file_path
            with open(file_path, 'r') as f:
                return json.load(f)
    
    def parse_model_output(self, log_file_or_list):
        """
        Parse your model's output from log file
        
        Expected format from your logs:
        "Table match: source_table -> target_table (conf: 0.XXX)"
        
        Args:
            log_file_or_list: Path to log file or list of prediction dicts
            
        Returns:
            list of dicts: [{'source': str, 'target': str, 'confidence': float}, ...]
        """
        predictions = []
        
        if isinstance(log_file_or_list, str):
            # Parse from log file
            with open(log_file_or_list, 'r') as f:
                for line in f:
                    if 'Table match:' in line:
                        # Extract source -> target (conf: X.XXX)
                        parts = line.split('Table match:')[1].strip()
                        match_part, conf_part = parts.split('(conf:')
                        source, target = match_part.split('->')
                        confidence = float(conf_part.strip().rstrip(')'))
                        
                        predictions.append({
                            'source': source.strip().lower(),
                            'target': target.strip().lower(),
                            'confidence': confidence
                        })
        else:
            # Already parsed predictions
            predictions = log_file_or_list
            
        return predictions
    
    def calculate_accuracy_at_k(self, predictions, k_values=[1, 3, 5]):
        """
        Calculate Accuracy@K (HitRate@K) - standard schema matching metric
        
        Args:
            predictions: list of prediction dicts with 'source', 'target', 'confidence'
            k_values: list of K values to evaluate
            
        Returns:
            dict: {k: accuracy_at_k}
        """
        # Group predictions by source table
        source_predictions = defaultdict(list)
        for pred in predictions:
            source_predictions[pred['source']].append(pred)
        
        # Sort by confidence for each source
        for source in source_predictions:
            source_predictions[source].sort(key=lambda x: x['confidence'], reverse=True)
        
        results = {}
        for k in k_values:
            hits = 0
            total = 0
            
            for source, gt_target in self.ground_truth.items():
                if source in source_predictions:
                    # Check if correct target is in top-K
                    top_k_targets = [p['target'] for p in source_predictions[source][:k]]
                    if gt_target in top_k_targets:
                        hits += 1
                total += 1
            
            results[f'Accuracy@{k}'] = hits / total if total > 0 else 0
        
        return results
    
    def calculate_confusion_matrix_metrics(self, predictions, confidence_threshold=0.5):
        """
        Calculate traditional classification metrics (Precision, Recall, F1)
        
        This treats schema matching as binary classification at table-pair level
        
        Args:
            predictions: list of prediction dicts
            confidence_threshold: threshold for considering a match positive
            
        Returns:
            dict: metrics including confusion matrix, precision, recall, F1
        """
        # Build all possible source-target pairs
        all_sources = set(self.ground_truth.keys())
        all_targets = set(self.ground_truth.values())
        
        # For each source, we have predictions
        y_true = []
        y_pred = []
        pair_labels = []
        
        # Build predictions dict for quick lookup
        pred_dict = defaultdict(list)
        for pred in predictions:
            if pred['confidence'] >= confidence_threshold:
                pred_dict[pred['source']].append(pred['target'])
        
        # For each source table in ground truth
        for source, gt_target in self.ground_truth.items():
            predicted_targets = pred_dict.get(source, [])
            
            # Check if correct match was predicted
            if gt_target in predicted_targets:
                y_true.append(1)  # True Positive
                y_pred.append(1)
                pair_labels.append(f"{source}->{gt_target}")
            else:
                y_true.append(1)  # Ground truth says yes
                y_pred.append(0)  # But we predicted no (False Negative)
                pair_labels.append(f"{source}->{gt_target}")
            
            # Add false positives (predicted but not in ground truth)
            for pred_target in predicted_targets:
                if pred_target != gt_target:
                    y_true.append(0)
                    y_pred.append(1)  # False Positive
                    pair_labels.append(f"{source}->{pred_target}")
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            'classification_report': classification_report(y_true, y_pred)
        }
    
    def evaluate(self, predictions, confidence_threshold=0.5, k_values=[1, 3, 5]):
        """
        Run full evaluation with both ranking and classification metrics
        
        Args:
            predictions: list of prediction dicts or path to log file
            confidence_threshold: for binary classification
            k_values: for Accuracy@K calculation
            
        Returns:
            dict: complete evaluation results
        """
        if isinstance(predictions, str):
            predictions = self.parse_model_output(predictions)
        
        # Calculate both types of metrics
        ranking_metrics = self.calculate_accuracy_at_k(predictions, k_values)
        classification_metrics = self.calculate_confusion_matrix_metrics(
            predictions, confidence_threshold
        )
        
        return {
            'ranking_metrics': ranking_metrics,
            'classification_metrics': classification_metrics,
            'num_ground_truth_mappings': len(self.ground_truth),
            'num_predictions': len(predictions)
        }
    
    def print_evaluation_report(self, results):
        """Print formatted evaluation report"""
        print("=" * 60)
        print("SCHEMA MATCHING EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\nDataset Info:")
        print(f"  Ground Truth Mappings: {results['num_ground_truth_mappings']}")
        print(f"  Total Predictions: {results['num_predictions']}")
        
        print(f"\n{'='*60}")
        print("RANKING METRICS (Standard for Schema Matching)")
        print(f"{'='*60}")
        for metric, value in results['ranking_metrics'].items():
            print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION METRICS")
        print(f"{'='*60}")
        cm_metrics = results['classification_metrics']
        print(f"  Precision: {cm_metrics['precision']:.4f}")
        print(f"  Recall: {cm_metrics['recall']:.4f}")
        print(f"  F1-Score: {cm_metrics['f1_score']:.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"  {'':>20} Predicted Negative  Predicted Positive")
        print(f"  Actual Negative:     {cm_metrics['true_negatives']:>6}            {cm_metrics['false_positives']:>6}")
        print(f"  Actual Positive:     {cm_metrics['false_negatives']:>6}            {cm_metrics['true_positives']:>6}")
        
        print(f"\n  True Positives (TP): {cm_metrics['true_positives']}")
        print(f"  False Positives (FP): {cm_metrics['false_positives']}")
        print(f"  False Negatives (FN): {cm_metrics['false_negatives']}")
        print(f"  True Negatives (TN): {cm_metrics['true_negatives']}")
        
        print(f"\n{'='*60}")


# Example usage
if __name__ == "__main__":
    # Example 1: From your log format
    example_predictions = [
        {'source': 'cohort_definition', 'target': 'patients', 'confidence': 0.751},
        {'source': 'provider', 'target': 'diagnoses_icd', 'confidence': 0.786},
        {'source': 'provider', 'target': 'admissions', 'confidence': 0.772},
        # Add more predictions...
    ]
    
    # Example ground truth (replace with actual from omop_mimic_data.xlsx)
    example_ground_truth = {
        'cohort_definition': 'cohort',
        'provider': 'caregivers',
        'person': 'patients',
        'visit_occurrence': 'admissions',
        # Add complete ground truth mappings...
    }
    
    # Initialize evaluator
    evaluator = SchemaMatchingEvaluator(example_ground_truth)
    
    # Run evaluation
    results = evaluator.evaluate(
        predictions=example_predictions,
        confidence_threshold=0.5,
        k_values=[1, 3, 5]
    )
    
    # Print report
    evaluator.print_evaluation_report(results)