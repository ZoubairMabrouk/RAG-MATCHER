#!/usr/bin/env python3
"""
Utilities for working with OMAP-MIMIC ground truth data

Features:
- Convert Excel ground truth to JSON
- Inspect and validate ground truth structure
- Generate statistics about the dataset
- Compare multiple ground truth versions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple
from collections import Counter

import pandas as pd


class OMAPGroundTruthUtils:
    """Utilities for OMAP ground truth data"""
    
    @staticmethod
    def convert_excel_to_json(
        excel_path: str, 
        output_path: str,
        sheet_name: int = 0
    ):
        """Convert Excel ground truth to JSON format"""
        print(f"Reading Excel file: {excel_path}")
        
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"Found columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        
        # Try to auto-detect columns
        columns = df.columns.tolist()
        source_col = columns[0]  # Typically OMOP tables
        target_col = columns[1]  # Typically MIMIC tables
        
        print(f"\nUsing columns:")
        print(f"  Source (OMOP): {source_col}")
        print(f"  Target (MIMIC): {target_col}")
        
        # Build mapping
        ground_truth = {}
        skipped = 0
        
        for idx, row in df.iterrows():
            source = str(row[source_col]).strip()
            target = str(row[target_col]).strip()
            
            # Skip invalid rows
            if source in ['nan', 'NaN', 'None', ''] or target in ['nan', 'NaN', 'None', '']:
                skipped += 1
                continue
            
            # Normalize to lowercase
            source = source.lower()
            target = target.lower()
            
            ground_truth[source] = target
        
        print(f"\nProcessed {len(ground_truth)} valid mappings ({skipped} skipped)")
        
        # Save to JSON
        output = {
            'metadata': {
                'source_file': excel_path,
                'source_column': source_col,
                'target_column': target_col,
                'num_mappings': len(ground_truth)
            },
            'mappings': ground_truth
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved to: {output_path}")
        return ground_truth
    
    @staticmethod
    def inspect_ground_truth(file_path: str, verbose: bool = False):
        """Inspect and validate ground truth structure"""
        
        print("=" * 80)
        print("OMAP GROUND TRUTH INSPECTION")
        print("=" * 80)
        
        # Load data
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            source_col = df.columns[0]
            target_col = df.columns[1]
            
            mappings = {}
            for _, row in df.iterrows():
                source = str(row[source_col]).strip().lower()
                target = str(row[target_col]).strip().lower()
                if source not in ['nan', ''] and target not in ['nan', '']:
                    mappings[source] = target
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
                mappings = data.get('mappings', data)
        
        # Statistics
        all_sources = list(mappings.keys())
        all_targets = list(mappings.values())
        
        print(f"\n{'Dataset Statistics':^80}")
        print("-" * 80)
        print(f"  Total Mappings: {len(mappings)}")
        print(f"  Unique Sources: {len(set(all_sources))}")
        print(f"  Unique Targets: {len(set(all_targets))}")
        
        # Check for duplicates
        source_counts = Counter(all_sources)
        target_counts = Counter(all_targets)
        
        duplicate_sources = {k: v for k, v in source_counts.items() if v > 1}
        duplicate_targets = {k: v for k, v in target_counts.items() if v > 1}
        
        if duplicate_sources:
            print(f"\n  ⚠ Warning: {len(duplicate_sources)} duplicate sources found:")
            for src, count in list(duplicate_sources.items())[:5]:
                print(f"    - {src}: {count} occurrences")
        
        if duplicate_targets:
            print(f"\n  Multiple sources map to same target:")
            for tgt, count in list(duplicate_targets.items())[:5]:
                sources = [s for s, t in mappings.items() if t == tgt]
                print(f"    - {tgt}: mapped from {sources[:3]}")
        
        # Mapping distribution
        print(f"\n{'Mapping Distribution':^80}")
        print("-" * 80)
        print(f"  Most Common Targets:")
        for target, count in target_counts.most_common(10):
            print(f"    {target:<30} ({count} sources)")
        
        if verbose:
            print(f"\n{'All Mappings':^80}")
            print("-" * 80)
            for source, target in sorted(mappings.items()):
                print(f"  {source:<35} -> {target}")
        
        return mappings
    
    @staticmethod
    def validate_consistency(ground_truth: Dict[str, str]) -> Dict[str, list]:
        """Validate consistency of ground truth mappings"""
        
        issues = {
            'ambiguous_sources': [],  # One source maps to multiple targets
            'many_to_one': [],  # Many sources map to one target
            'naming_issues': []  # Potential naming inconsistencies
        }
        
        # Check for ambiguous sources (shouldn't exist in properly formatted GT)
        source_to_targets = {}
        for source, target in ground_truth.items():
            if source in source_to_targets:
                issues['ambiguous_sources'].append(
                    (source, [source_to_targets[source], target])
                )
            source_to_targets[source] = target
        
        # Check for many-to-one mappings
        target_to_sources = {}
        for source, target in ground_truth.items():
            if target not in target_to_sources:
                target_to_sources[target] = []
            target_to_sources[target].append(source)
        
        for target, sources in target_to_sources.items():
            if len(sources) > 1:
                issues['many_to_one'].append((target, sources))
        
        # Check naming conventions
        for source in ground_truth.keys():
            # Check for mixed case or special characters
            if source != source.lower():
                issues['naming_issues'].append(
                    f"Mixed case in source: {source}"
                )
            if any(c in source for c in [' ', '-', '.']):
                issues['naming_issues'].append(
                    f"Special characters in source: {source}"
                )
        
        return issues
    
    @staticmethod
    def generate_statistics_report(ground_truth: Dict[str, str], output_file: str = None):
        """Generate comprehensive statistics report"""
        
        report = {
            'summary': {
                'total_mappings': len(ground_truth),
                'unique_sources': len(set(ground_truth.keys())),
                'unique_targets': len(set(ground_truth.values()))
            },
            'target_distribution': {},
            'source_patterns': {},
            'consistency_check': {}
        }
        
        # Target distribution
        target_counts = Counter(ground_truth.values())
        report['target_distribution'] = dict(target_counts.most_common())
        
        # Source patterns (prefixes, suffixes)
        prefixes = Counter()
        suffixes = Counter()
        for source in ground_truth.keys():
            parts = source.split('_')
            if len(parts) > 1:
                prefixes[parts[0]] += 1
                suffixes[parts[-1]] += 1
        
        report['source_patterns'] = {
            'common_prefixes': dict(prefixes.most_common(10)),
            'common_suffixes': dict(suffixes.most_common(10))
        }
        
        # Consistency check
        issues = OMAPGroundTruthUtils.validate_consistency(ground_truth)
        report['consistency_check'] = {
            'ambiguous_sources_count': len(issues['ambiguous_sources']),
            'many_to_one_count': len(issues['many_to_one']),
            'naming_issues_count': len(issues['naming_issues'])
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Statistics report saved to: {output_file}")
        
        return report
    
    @staticmethod
    def create_sample_ground_truth(output_file: str):
        """Create a sample ground truth file for testing"""
        
        sample_mappings = {
            'person': 'patients',
            'visit_occurrence': 'admissions',
            'condition_occurrence': 'diagnoses_icd',
            'procedure_occurrence': 'procedures_icd',
            'drug_exposure': 'prescriptions',
            'measurement': 'labevents',
            'observation': 'chartevents',
            'care_site': 'services',
            'provider': 'caregivers',
            'death': 'patients',
            'cohort_definition': 'cohort',
            'note': 'noteevents'
        }
        
        output = {
            'metadata': {
                'description': 'Sample OMOP to MIMIC-III table mappings',
                'source': 'Generated for testing',
                'num_mappings': len(sample_mappings)
            },
            'mappings': sample_mappings
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Created sample ground truth: {output_file}")
        print(f"Contains {len(sample_mappings)} mappings")


def main():
    parser = argparse.ArgumentParser(
        description='OMAP Ground Truth Utilities'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert Excel ground truth to JSON'
    )
    convert_parser.add_argument(
        'input',
        help='Input Excel file'
    )
    convert_parser.add_argument(
        'output',
        help='Output JSON file'
    )
    convert_parser.add_argument(
        '--sheet',
        type=int,
        default=0,
        help='Sheet index (default: 0)'
    )
    
    # Inspect command
    inspect_parser = subparsers.add_parser(
        'inspect',
        help='Inspect ground truth structure'
    )
    inspect_parser.add_argument(
        'file',
        help='Ground truth file (Excel or JSON)'
    )
    inspect_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show all mappings'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        help='Generate statistics report'
    )
    stats_parser.add_argument(
        'file',
        help='Ground truth file'
    )
    stats_parser.add_argument(
        '--output',
        help='Output file for report'
    )
    
    # Sample command
    sample_parser = subparsers.add_parser(
        'sample',
        help='Create sample ground truth for testing'
    )
    sample_parser.add_argument(
        'output',
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        OMAPGroundTruthUtils.convert_excel_to_json(
            args.input,
            args.output,
            sheet_name=args.sheet
        )
    
    elif args.command == 'inspect':
        OMAPGroundTruthUtils.inspect_ground_truth(
            args.file,
            verbose=args.verbose
        )
    
    elif args.command == 'stats':
        # Load ground truth
        if args.file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.file)
            mappings = {
                str(row[df.columns[0]]).strip().lower(): str(row[df.columns[1]]).strip().lower()
                for _, row in df.iterrows()
                if str(row[df.columns[0]]) != 'nan' and str(row[df.columns[1]]) != 'nan'
            }
        else:
            with open(args.file, 'r') as f:
                data = json.load(f)
                mappings = data.get('mappings', data)
        
        OMAPGroundTruthUtils.generate_statistics_report(
            mappings,
            output_file=args.output
        )
    
    elif args.command == 'sample':
        OMAPGroundTruthUtils.create_sample_ground_truth(args.output)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 