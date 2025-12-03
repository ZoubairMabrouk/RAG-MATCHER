#!/usr/bin/env python3
"""
Grid search over table and column thresholds for RAG schema matcher

This script runs the RAG matcher with different threshold combinations
and evaluates each against the OMAP-MIMIC ground truth to find optimal parameters.

Usage:
------
python evaluation/scripts/claude/adv/grid_search.py \
    --uschema evaluation/data/omap/uschema_testdata.json \
    --ground-truth evaluation/data/omap/omop_mimic_data.xlsx \
    --table-thresholds 0.0 0.2 0.4 0.6 \
    --column-thresholds 0.0 0.2 0.4 0.6 \
    --output results/grid_search.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("threshold_grid_search")


class ThresholdGridSearch:
    """Grid search over RAG matcher thresholds"""
    
    def __init__(
        self,
        uschema_file: str,
        ground_truth_file: str,
        db_url: str = None,
        dialect: str = "postgresql",
        kb_file: str = None
    ):
        self.uschema_file = uschema_file
        self.ground_truth_file = ground_truth_file
        self.db_url = db_url or "postgresql://test:test@localhost:55432/test"
        self.dialect = dialect
        self.kb_file = kb_file
        
        # Create output directory
        self.output_dir = Path("artifacts/grid_search")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def run_matching(
        self,
        table_threshold: float,
        column_threshold: float,
        run_id: str
    ) -> str:
        """
        Run RAG matching with specific thresholds
        
        Returns:
            Path to output JSON file
        """
        output_file = self.output_dir / f"matching_{run_id}.json"
        
        cmd = [
            "python", "scripts/run_rag_virtual_rename.py",
            "--uschema-file", self.uschema_file,
            "--db-url", self.db_url,
            "--dialect", self.dialect,
            "--table-threshold", str(table_threshold),
            "--column-threshold", str(column_threshold),
            "--top-k", "5",
            "--out", str(output_file)
        ]
        
        if self.kb_file:
            cmd.extend(["--kb-file", self.kb_file])
        
        log.info(f"Running: table_th={table_threshold}, column_th={column_threshold}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return str(output_file)
        except subprocess.CalledProcessError as e:
            log.error(f"Matching failed: {e.stderr}")
            return None
    
    def evaluate_predictions(
        self,
        predictions_file: str,
        eval_threshold: float = 0.5
    ) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Returns:
            Dict with evaluation metrics
        """
        cmd = [
            "python", "scripts/evaluate_omap_column_matching.py",
            "--ground-truth", self.ground_truth_file,
            "--predictions", predictions_file,
            "--threshold", str(eval_threshold)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output to extract metrics
            # (In practice, you'd modify the evaluation script to output JSON)
            # For now, we'll parse from the predictions file and evaluate directly
            
            from evaluation.scripts.claude.adv.evaluator import OMAPColumnEvaluator
            
            evaluator = OMAPColumnEvaluator(self.ground_truth_file)
            predictions = evaluator.parse_rag_predictions(predictions_file)
            metrics = evaluator.evaluate_at_threshold(predictions, eval_threshold)
            
            return metrics
            
        except Exception as e:
            log.error(f"Evaluation failed: {e}")
            return {'error': str(e)}
    
    def grid_search(
        self,
        table_thresholds: List[float],
        column_thresholds: List[float],
        eval_thresholds: List[float] = [0.5]
    ) -> pd.DataFrame:
        """
        Perform grid search over threshold combinations
        
        Args:
            table_thresholds: List of table matching thresholds to try
            column_thresholds: List of column matching thresholds to try
            eval_thresholds: List of evaluation thresholds for predictions
            
        Returns:
            DataFrame with results
        """
        total_runs = len(table_thresholds) * len(column_thresholds) * len(eval_thresholds)
        log.info(f"Starting grid search: {total_runs} total configurations")
        
        run_counter = 0
        
        for table_th in table_thresholds:
            for column_th in column_thresholds:
                run_counter += 1
                run_id = f"t{table_th:.2f}_c{column_th:.2f}"
                
                log.info(f"\n{'='*60}")
                log.info(f"Run {run_counter}/{total_runs}")
                log.info(f"Table threshold: {table_th}, Column threshold: {column_th}")
                log.info(f"{'='*60}")
                
                # Run matching
                predictions_file = self.run_matching(table_th, column_th, run_id)
                
                if not predictions_file:
                    continue
                
                # Evaluate at different eval thresholds
                for eval_th in eval_thresholds:
                    metrics = self.evaluate_predictions(predictions_file, eval_th)
                    
                    if 'error' not in metrics:
                        result = {
                            'run_id': run_id,
                            'table_threshold': table_th,
                            'column_threshold': column_th,
                            'eval_threshold': eval_th,
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1_score': metrics['f1_score'],
                            'accuracy': metrics['accuracy'],
                            'true_positives': metrics['true_positives'],
                            'false_positives': metrics['false_positives'],
                            'false_negatives': metrics['false_negatives'],
                            'total_predictions': metrics['total_predictions'],
                            'predictions_file': predictions_file
                        }
                        
                        self.results.append(result)
                        
                        log.info(f"  Eval threshold {eval_th}: "
                                f"P={metrics['precision']:.3f}, "
                                f"R={metrics['recall']:.3f}, "
                                f"F1={metrics['f1_score']:.3f}")
        
        return pd.DataFrame(self.results)
    
    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """
        Analyze grid search results
        
        Returns:
            Dict with analysis summary
        """
        if df.empty:
            return {'error': 'No results to analyze'}
        
        # Find best configuration by F1-score
        best_idx = df['f1_score'].idxmax()
        best_config = df.loc[best_idx].to_dict()
        
        # Find best by precision
        best_precision_idx = df['precision'].idxmax()
        best_precision = df.loc[best_precision_idx].to_dict()
        
        # Find best by recall
        best_recall_idx = df['recall'].idxmax()
        best_recall = df.loc[best_recall_idx].to_dict()
        
        # Aggregate statistics
        analysis = {
            'best_f1': best_config,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'summary_statistics': {
                'mean_f1': float(df['f1_score'].mean()),
                'std_f1': float(df['f1_score'].std()),
                'mean_precision': float(df['precision'].mean()),
                'mean_recall': float(df['recall'].mean()),
                'max_f1': float(df['f1_score'].max()),
                'min_f1': float(df['f1_score'].min())
            },
            'threshold_sensitivity': self._analyze_threshold_sensitivity(df)
        }
        
        return analysis
    
    def _analyze_threshold_sensitivity(self, df: pd.DataFrame) -> Dict:
        """Analyze how sensitive metrics are to threshold changes"""
        sensitivity = {}
        
        # Table threshold sensitivity
        if len(df['table_threshold'].unique()) > 1:
            grouped = df.groupby('table_threshold')['f1_score'].mean()
            sensitivity['table_threshold'] = {
                'values': grouped.index.tolist(),
                'mean_f1': grouped.values.tolist(),
                'variance': float(grouped.var())
            }
        
        # Column threshold sensitivity
        if len(df['column_threshold'].unique()) > 1:
            grouped = df.groupby('column_threshold')['f1_score'].mean()
            sensitivity['column_threshold'] = {
                'values': grouped.index.tolist(),
                'mean_f1': grouped.values.tolist(),
                'variance': float(grouped.var())
            }
        
        return sensitivity
    
    def print_report(self, df: pd.DataFrame, analysis: Dict):
        """Print comprehensive grid search report"""
        print("\n" + "=" * 80)
        print("GRID SEARCH RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal configurations tested: {len(df)}")
        
        # Best configuration
        best = analysis['best_f1']
        print(f"\n{'BEST CONFIGURATION (by F1-score)':^80}")
        print("-" * 80)
        print(f"  Table Threshold:    {best['table_threshold']:.2f}")
        print(f"  Column Threshold:   {best['column_threshold']:.2f}")
        print(f"  Eval Threshold:     {best['eval_threshold']:.2f}")
        print(f"  F1-Score:           {best['f1_score']:.4f}")
        print(f"  Precision:          {best['precision']:.4f}")
        print(f"  Recall:             {best['recall']:.4f}")
        print(f"  Accuracy:           {best['accuracy']:.4f}")
        print(f"  True Positives:     {best['true_positives']}")
        print(f"  False Positives:    {best['false_positives']}")
        print(f"  False Negatives:    {best['false_negatives']}")
        
        # Alternative configurations
        print(f"\n{'ALTERNATIVE CONFIGURATIONS':^80}")
        print("-" * 80)
        
        best_prec = analysis['best_precision']
        print(f"Best Precision: {best_prec['precision']:.4f} "
              f"(table={best_prec['table_threshold']:.2f}, "
              f"column={best_prec['column_threshold']:.2f})")
        
        best_rec = analysis['best_recall']
        print(f"Best Recall:    {best_rec['recall']:.4f} "
              f"(table={best_rec['table_threshold']:.2f}, "
              f"column={best_rec['column_threshold']:.2f})")
        
        # Summary statistics
        stats = analysis['summary_statistics']
        print(f"\n{'SUMMARY STATISTICS':^80}")
        print("-" * 80)
        print(f"  Mean F1-Score:      {stats['mean_f1']:.4f} (±{stats['std_f1']:.4f})")
        print(f"  F1-Score Range:     {stats['min_f1']:.4f} - {stats['max_f1']:.4f}")
        print(f"  Mean Precision:     {stats['mean_precision']:.4f}")
        print(f"  Mean Recall:        {stats['mean_recall']:.4f}")
        
        # Threshold sensitivity
        if 'threshold_sensitivity' in analysis:
            sens = analysis['threshold_sensitivity']
            print(f"\n{'THRESHOLD SENSITIVITY':^80}")
            print("-" * 80)
            
            if 'table_threshold' in sens:
                print(f"  Table threshold variance: {sens['table_threshold']['variance']:.4f}")
            
            if 'column_threshold' in sens:
                print(f"  Column threshold variance: {sens['column_threshold']['variance']:.4f}")
        
        # Top 5 configurations
        print(f"\n{'TOP 5 CONFIGURATIONS':^80}")
        print("-" * 80)
        top5 = df.nlargest(5, 'f1_score')[
            ['table_threshold', 'column_threshold', 'eval_threshold', 
             'f1_score', 'precision', 'recall']
        ]
        print(top5.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for optimal RAG matcher thresholds"
    )
    
    parser.add_argument(
        '--uschema',
        required=True,
        help='Path to U-Schema JSON file'
    )
    parser.add_argument(
        '--ground-truth',
        required=True,
        help='Path to OMAP-MIMIC ground truth (omop_mimic_data.xlsx)'
    )
    parser.add_argument(
        '--db-url',
        help='Database URL (default: postgresql://test:test@localhost:55432/test)'
    )
    parser.add_argument(
        '--table-thresholds',
        nargs='+',
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
        help='Table matching thresholds to try'
    )
    parser.add_argument(
        '--column-thresholds',
        nargs='+',
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
        help='Column matching thresholds to try'
    )
    parser.add_argument(
        '--eval-thresholds',
        nargs='+',
        type=float,
        default=[0.5],
        help='Evaluation thresholds for predictions'
    )
    parser.add_argument(
        '--kb-file',
        help='External knowledge base file'
    )
    parser.add_argument(
        '--output',
        default='artifacts/grid_search_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Initialize grid search
    grid_search = ThresholdGridSearch(
        uschema_file=args.uschema,
        ground_truth_file=args.ground_truth,
        db_url=args.db_url,
        kb_file=args.kb_file
    )
    
    # Run grid search
    log.info("Starting grid search...")
    results_df = grid_search.grid_search(
        table_thresholds=args.table_thresholds,
        column_thresholds=args.column_thresholds,
        eval_thresholds=args.eval_thresholds
    )
    
    if results_df.empty:
        log.error("No results generated!")
        return 1
    
    # Analyze results
    analysis = grid_search.analyze_results(results_df)
    
    # Print report
    grid_search.print_report(results_df, analysis)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'uschema_file': args.uschema,
            'ground_truth_file': args.ground_truth,
            'table_thresholds': args.table_thresholds,
            'column_thresholds': args.column_thresholds,
            'eval_thresholds': args.eval_thresholds
        },
        'results': results_df.to_dict('records'),
        'analysis': analysis
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    log.info(f"\nResults saved to: {output_path}")
    
    # Also save CSV for easy viewing
    csv_path = output_path.with_suffix('.csv')
    results_df.to_csv(csv_path, index=False)
    log.info(f"CSV saved to: {csv_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())