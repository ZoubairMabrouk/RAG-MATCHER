# OMAP-MIMIC Column-Level Evaluation Guide

## Overview

This guide shows you how to evaluate your RAG schema matcher against the OMAP-MIMIC dataset, which contains **64,080 column-level pairs** (129 positive matches, 63,951 negative pairs).

## Dataset Structure

The OMAP-MIMIC dataset (`omop_mimic_data.xlsx`) has this format:

| Column | Description | Example |
|--------|-------------|---------|
| `omop` | OMOP table-column pair | `person-person_id` |
| `table` | MIMIC table-column pair | `patients-subject_id` |
| `label` | 1 if match, 0 if not | `1` or `0` |

**Key Statistics:**
- Total pairs: 64,080
- Positive matches (label=1): 129 (~0.2%)
- Negative pairs (label=0): 63,951 (~99.8%)
- **Highly imbalanced** dataset

## Quick Start

### Option 1: Run Complete Evaluation (Recommended)

```bash
# Make script executable
chmod +x scripts/run_omap_evaluation.sh

# Run with default settings
./scripts/run_omap_evaluation.sh
```

This automatically:
1. Tests threshold combinations (table: 0.0-0.8, column: 0.0-0.8)
2. Finds best configuration
3. Runs detailed evaluation
4. Generates comprehensive reports

### Option 2: Custom Grid Search

```bash
python scripts/threshold_grid_search.py \
    --uschema uschema_testdata.json \
    --ground-truth omop_mimic_data.xlsx \
    --table-thresholds 0.0 0.1 0.2 0.3 0.4 0.5 \
    --column-thresholds 0.0 0.1 0.2 0.3 0.4 0.5 \
    --output results/grid_search.json
```

### Option 3: Evaluate Single Configuration

```bash
# Step 1: Run matching with specific thresholds
python scripts/run_rag_virtual_rename.py \
    --uschema-file uschema_testdata.json \
    --db-url postgresql://test:test@localhost:55432/test \
    --table-threshold 0.4 \
    --column-threshold 0.3 \
    --out artifacts/predictions.json

# Step 2: Evaluate
python scripts/evaluate_omap_column_matching.py \
    --ground-truth omop_mimic_data.xlsx \
    --predictions artifacts/predictions.json \
    --threshold 0.5 \
    --verbose
```

## Understanding the Metrics

### Primary Metrics

**Precision**: Of the column pairs you predicted as matches, what percentage were actually matches?
- Formula: TP / (TP + FP)
- **High precision** = Few false matches
- Use case: When you need to be confident in your predictions

**Recall**: Of all the actual column matches, what percentage did you find?
- Formula: TP / (TP + FN)  
- **High recall** = Few missed matches
- Use case: When you need to find all possible matches

**F1-Score**: Harmonic mean of precision and recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- **Best overall metric** for imbalanced datasets
- Use this to find optimal threshold

### Example Results Interpretation

```
Metrics at threshold 0.5:
  Precision: 0.8500  (85% of predicted matches are correct)
  Recall:    0.6200  (Found 62% of all true matches)
  F1-Score:  0.7179  (Good balance)
  Accuracy:  0.9985  (High due to many true negatives)
```

**What this means:**
- Your model is **conservative** (high precision, lower recall)
- When it predicts a match, it's usually right (85%)
- But it misses some matches (38% false negative rate)
- **Action**: Lower threshold to improve recall

## Threshold Selection Strategy

### Understanding Thresholds

You have **three** threshold types:

1. **Table Threshold** (`--table-threshold`): Minimum confidence to match tables
   - Lower → More table matches → More columns to evaluate
   - Higher → Fewer table matches → Fewer columns evaluated

2. **Column Threshold** (`--column-threshold`): Minimum confidence to match columns
   - Lower → More column matches → Higher recall, lower precision
   - Higher → Fewer column matches → Lower recall, higher precision

3. **Evaluation Threshold**: Threshold for counting a prediction as "positive"
   - Used during evaluation only, doesn't affect matching

### Recommended Starting Points

**For high precision** (few false positives):
```bash
--table-threshold 0.6 \
--column-threshold 0.6
```

**For high recall** (few false negatives):
```bash
--table-threshold 0.2 \
--column-threshold 0.2
```

**For balanced F1** (use grid search to find):
```bash
# Grid search will find optimal balance
--table-thresholds 0.2 0.3 0.4 0.5 0.6 \
--column-thresholds 0.2 0.3 0.4 0.5 0.6
```

## Working with Imbalanced Data

The OMAP dataset is **highly imbalanced** (99.8% negative pairs). Here's how to handle it:

### 1. Don't Rely on Accuracy Alone

```python
# Accuracy will be high even for bad models!
# Even predicting "no match" for everything gives 99.8% accuracy

# Always look at:
- Precision
- Recall  
- F1-Score
- Confusion Matrix
```

### 2. Focus on Positive Class Metrics

```python
# What matters:
True Positives:  How many correct matches you found
False Positives: How many wrong matches you predicted
False Negatives: How many matches you missed

# Less important for imbalanced data:
True Negatives:  Usually very high due to many negative pairs
Accuracy:        Misleading when classes are imbalanced
```

### 3. Use F1-Score for Optimization

The grid search automatically uses F1-score to find the best configuration.

## Interpreting Results

### Good Performance Indicators

✅ **F1-Score > 0.70**: Your model is working well
✅ **Precision > 0.80**: Predictions are reliable
✅ **Recall > 0.60**: Finding most matches
✅ **Low FP rate**: Not making many wrong predictions

### Warning Signs

⚠️ **F1-Score < 0.50**: Model needs improvement
⚠️ **Very low recall** (< 0.30): Missing too many matches → Lower thresholds
⚠️ **Very low precision** (< 0.50): Too many false matches → Raise thresholds
⚠️ **Many FP on similar tables**: May need better semantic understanding

### Example Analysis

```
Configuration: table_th=0.4, column_th=0.3

Results:
  F1-Score:  0.7234
  Precision: 0.8512
  Recall:    0.6279
  
True Positives:  81 / 129 matches found
False Positives: 14 / 95 total predictions
False Negatives: 48 / matches missed

Analysis:
✅ Good precision - When model predicts match, usually correct
⚠️  Moderate recall - Missing ~38% of matches
💡 Action: Try lower column threshold (0.2) to improve recall
```

## Output Files

### Grid Search Results

**`grid_search_results.json`**: Complete results
```json
{
  "analysis": {
    "best_f1": {
      "table_threshold": 0.4,
      "column_threshold": 0.3,
      "f1_score": 0.7234,
      ...
    }
  },
  "results": [
    {
      "table_threshold": 0.0,
      "column_threshold": 0.0,
      "precision": 0.6234,
      "recall": 0.8914,
      "f1_score": 0.7334
    },
    ...
  ]
}
```

**`grid_search_results.csv`**: Easy to analyze in Excel
```csv
table_threshold,column_threshold,precision,recall,f1_score,accuracy
0.0,0.0,0.6234,0.8914,0.7334,0.9978
0.0,0.2,0.7123,0.7829,0.7459,0.9982
...
```

### Evaluation Results

**`eval_0.5.json`**: Detailed evaluation at specific threshold
```json
{
  "threshold": 0.5,
  "precision": 0.8512,
  "recall": 0.6279,
  "f1_score": 0.7234,
  "confusion_matrix": [[63937, 14], [48, 81]],
  "examples": {
    "true_positives": [
      {
        "omop": "person-person_id",
        "mimic": "patients-subject_id",
        "confidence": 0.92
      },
      ...
    ],
    "false_positives": [...],
    "false_negatives": [...]
  }
}
```

## Troubleshooting

### Issue: All predictions have low confidence

**Symptoms:**
- Most confidences < 0.3
- Few matches even at low thresholds

**Solutions:**
1. Check your knowledge base enrichment
2. Verify embedding quality
3. Review table/column descriptions

```bash
# Try with enriched KB
python scripts/run_rag_virtual_rename.py \
    --kb-file data/rag/knowledge_base_enriched.jsonl \
    ...
```

### Issue: Too many false positives

**Symptoms:**
- Low precision (<0.5)
- Many incorrect matches

**Solutions:**
1. Increase thresholds
2. Improve semantic matching
3. Add negative examples to KB

```bash
# More conservative thresholds
--table-threshold 0.6 \
--column-threshold 0.6
```

### Issue: Missing many matches

**Symptoms:**
- Low recall (<0.5)
- Many false negatives

**Solutions:**
1. Lower thresholds
2. Increase top-k retrieval
3. Improve embeddings

```bash
# More aggressive matching
--table-threshold 0.2 \
--column-threshold 0.2 \
--top-k 10
```

### Issue: Grid search takes too long

**Solution:** Reduce search space

```bash
# Coarse search first
--table-thresholds 0.0 0.3 0.6 0.9 \
--column-thresholds 0.0 0.3 0.6 0.9

# Then fine-tune around best
--table-thresholds 0.25 0.30 0.35 \
--column-thresholds 0.25 0.30 0.35
```

## Advanced Usage

### Custom Evaluation Metrics

Modify `evaluate_omap_column_matching.py` to add:

```python
# Add weighted F1-score (gives more weight to recall)
from sklearn.metrics import fbeta_score

# F2-score (recall weighted 2x more than precision)
f2 = fbeta_score(y_true, y_pred, beta=2.0)

# F0.5-score (precision weighted 2x more than recall)  
f05 = fbeta_score(y_true, y_pred, beta=0.5)
```

### Analyzing Specific Table Pairs

```python
# In evaluation script, filter by table:
person_results = [
    r for r in predictions 
    if r['omop_table'] == 'person'
]

# Evaluate only person table matching
person_metrics = evaluator.evaluate_at_threshold(person_results, 0.5)
```

### Cross-Validation

For more robust evaluation:

```python
# Split OMAP dataset into train/val/test
# Train: Use to tune KB enrichment
# Val: Use for threshold selection (grid search)
# Test: Final evaluation (held out)
```

## Best Practices

1. **Always run grid search first** to find optimal thresholds
2. **Use F1-score** as primary metric for imbalanced data
3. **Examine false positives/negatives** to understand errors
4. **Test at multiple eval thresholds** (0.3, 0.5, 0.7)
5. **Save all configurations** for reproducibility
6. **Compare with baselines** (string similarity, etc.)

## Expected Performance

Based on schema matching literature:

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| F1-Score | <0.50 | 0.50-0.65 | 0.65-0.80 | >0.80 |
| Precision | <0.60 | 0.60-0.75 | 0.75-0.90 | >0.90 |
| Recall | <0.50 | 0.50-0.70 | 0.70-0.85 | >0.85 |

For the OMAP-MIMIC dataset:
- **Good performance**: F1 > 0.65
- **Excellent performance**: F1 > 0.75

## Next Steps

After evaluation:

1. **If results are good** (F1 > 0.70):
   - Document best configuration
   - Test on other datasets
   - Deploy to production

2. **If results are moderate** (F1: 0.50-0.70):
   - Enrich knowledge base
   - Improve embeddings
   - Tune thresholds more finely

3. **If results are poor** (F1 < 0.50):
   - Review matching logic
   - Check data quality
   - Consider hybrid approaches

## Support

For issues:
1. Check logs in `artifacts/grid_search/*.log`
2. Examine failed predictions
3. Review confusion matrix examples
4. Compare with ground truth manually