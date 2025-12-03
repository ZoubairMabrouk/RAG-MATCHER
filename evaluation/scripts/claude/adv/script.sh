#!/bin/bash
# Complete evaluation workflow for RAG schema matcher on OMAP-MIMIC dataset
#
# This script performs:
# 1. Grid search across threshold combinations
# 2. Detailed evaluation at best threshold
# 3. Generation of comprehensive reports

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
USCHEMA_FILE="${USCHEMA_FILE:-uschema_testdata.json}"
GROUND_TRUTH="${GROUND_TRUTH:-omop_mimic_data.xlsx}"
DB_URL="${DATABASE_URL:-postgresql://test:test@localhost:55432/test}"
OUTPUT_DIR="artifacts/omap_evaluation_$(date +%Y%m%d_%H%M%S)"

# Threshold ranges
TABLE_THRESHOLDS="${TABLE_THRESHOLDS:-0.0 0.2 0.4 0.6 0.8}"
COLUMN_THRESHOLDS="${COLUMN_THRESHOLDS:-0.0 0.2 0.4 0.6 0.8}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OMAP-MIMIC Evaluation Workflow${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
echo -e "${GREEN}Output directory: ${OUTPUT_DIR}${NC}\n"

# Check files exist
echo -e "${YELLOW}[1/4] Checking input files...${NC}"

if [ ! -f "${USCHEMA_FILE}" ]; then
    echo -e "${RED}Error: U-Schema file not found: ${USCHEMA_FILE}${NC}"
    exit 1
fi

if [ ! -f "${GROUND_TRUTH}" ]; then
    echo -e "${RED}Error: Ground truth file not found: ${GROUND_TRUTH}${NC}"
    echo "Download from: https://github.com/JZCS2018/SMAT/raw/main/datasets/omap/omop_mimic_data.xlsx"
    exit 1
fi

echo -e "${GREEN}✓ Input files found${NC}\n"

# Step 2: Run grid search
echo -e "${YELLOW}[2/4] Running grid search...${NC}"
echo "Table thresholds: ${TABLE_THRESHOLDS}"
echo "Column thresholds: ${COLUMN_THRESHOLDS}"
echo ""

python scripts/threshold_grid_search.py \
    --uschema "${USCHEMA_FILE}" \
    --ground-truth "${GROUND_TRUTH}" \
    --db-url "${DB_URL}" \
    --table-thresholds ${TABLE_THRESHOLDS} \
    --column-thresholds ${COLUMN_THRESHOLDS} \
    --output "${OUTPUT_DIR}/grid_search_results.json"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Grid search failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Grid search complete${NC}\n"

# Step 3: Extract best configuration and run detailed evaluation
echo -e "${YELLOW}[3/4] Running detailed evaluation at best threshold...${NC}"

# Extract best thresholds from grid search results
BEST_TABLE=$(python3 << 'EOF'
import json
with open("${OUTPUT_DIR}/grid_search_results.json") as f:
    data = json.load(f)
    print(data['analysis']['best_f1']['table_threshold'])
EOF
)

BEST_COLUMN=$(python3 << 'EOF'
import json
with open("${OUTPUT_DIR}/grid_search_results.json") as f:
    data = json.load(f)
    print(data['analysis']['best_f1']['column_threshold'])
EOF
)

echo "Best table threshold: ${BEST_TABLE}"
echo "Best column threshold: ${BEST_COLUMN}"
echo ""

# Run matching with best thresholds
python scripts/run_rag_virtual_rename.py \
    --uschema-file "${USCHEMA_FILE}" \
    --db-url "${DB_URL}" \
    --table-threshold "${BEST_TABLE}" \
    --column-threshold "${BEST_COLUMN}" \
    --out "${OUTPUT_DIR}/best_matching.json"

# Evaluate with multiple eval thresholds
for eval_th in 0.3 0.4 0.5 0.6 0.7; do
    echo "Evaluating at threshold ${eval_th}..."
    python scripts/evaluate_omap_column_matching.py \
        --ground-truth "${GROUND_TRUTH}" \
        --predictions "${OUTPUT_DIR}/best_matching.json" \
        --threshold ${eval_th} \
        --output "${OUTPUT_DIR}/eval_${eval_th}.json" \
        --verbose
done

echo -e "${GREEN}✓ Detailed evaluation complete${NC}\n"

# Step 4: Generate summary report
echo -e "${YELLOW}[4/4] Generating summary report...${NC}"

python3 << EOF
import json
import pandas as pd
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")

# Load grid search results
with open(output_dir / "grid_search_results.json") as f:
    grid_data = json.load(f)

# Create summary report
report = []
report.append("=" * 80)
report.append("OMAP-MIMIC EVALUATION SUMMARY REPORT")
report.append("=" * 80)
report.append("")

# Best configuration
best = grid_data['analysis']['best_f1']
report.append("BEST CONFIGURATION:")
report.append(f"  Table Threshold:    {best['table_threshold']:.2f}")
report.append(f"  Column Threshold:   {best['column_threshold']:.2f}")
report.append(f"  Evaluation Threshold: {best['eval_threshold']:.2f}")
report.append("")
report.append("  Metrics:")
report.append(f"    F1-Score:         {best['f1_score']:.4f}")
report.append(f"    Precision:        {best['precision']:.4f}")
report.append(f"    Recall:           {best['recall']:.4f}")
report.append(f"    Accuracy:         {best['accuracy']:.4f}")
report.append("")
report.append(f"  Matches:")
report.append(f"    True Positives:   {best['true_positives']}")
report.append(f"    False Positives:  {best['false_positives']}")
report.append(f"    False Negatives:  {best['false_negatives']}")
report.append("")

# Grid search summary
stats = grid_data['analysis']['summary_statistics']
report.append("GRID SEARCH SUMMARY:")
report.append(f"  Configurations tested: {len(grid_data['results'])}")
report.append(f"  Mean F1-Score:        {stats['mean_f1']:.4f} (±{stats['std_f1']:.4f})")
report.append(f"  F1-Score Range:       {stats['min_f1']:.4f} - {stats['max_f1']:.4f}")
report.append("")

# Top 5 configurations
report.append("TOP 5 CONFIGURATIONS:")
df = pd.DataFrame(grid_data['results'])
top5 = df.nlargest(5, 'f1_score')
for idx, row in top5.iterrows():
    report.append(f"  {row['table_threshold']:.2f}/{row['column_threshold']:.2f} -> "
                 f"F1={row['f1_score']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}")
report.append("")

# Files generated
report.append("OUTPUT FILES:")
report.append(f"  Grid search results:  grid_search_results.json")
report.append(f"  Best matching:        best_matching.json")
report.append(f"  Evaluation results:   eval_*.json")
report.append("")

report.append("=" * 80)

# Save report
report_text = "\n".join(report)
print(report_text)

with open(output_dir / "SUMMARY_REPORT.txt", "w") as f:
    f.write(report_text)

print(f"\nReport saved to: {output_dir}/SUMMARY_REPORT.txt")
EOF

echo -e "${GREEN}✓ Summary report generated${NC}\n"

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo "All results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - SUMMARY_REPORT.txt       : Quick summary of results"
echo "  - grid_search_results.json : Complete grid search data"
echo "  - grid_search_results.csv  : Results in CSV format"
echo "  - best_matching.json       : Predictions at best threshold"
echo "  - eval_*.json              : Detailed evaluations"
echo ""

# Display best configuration
echo -e "${GREEN}Best Configuration:${NC}"
python3 << EOF
import json
with open("${OUTPUT_DIR}/grid_search_results.json") as f:
    data = json.load(f)
    best = data['analysis']['best_f1']
    print(f"  Table Threshold:   {best['table_threshold']:.2f}")
    print(f"  Column Threshold:  {best['column_threshold']:.2f}")
    print(f"  F1-Score:          {best['f1_score']:.4f}")
    print(f"  Precision:         {best['precision']:.4f}")
    print(f"  Recall:            {best['recall']:.4f}")
EOF

echo ""
echo "View full report: cat ${OUTPUT_DIR}/SUMMARY_REPORT.txt"
echo "View CSV:         open ${OUTPUT_DIR}/grid_search_results.csv"