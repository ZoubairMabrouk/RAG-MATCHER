#{"variant":"standard","id":"27384"}
#python
# Required packages: pandas, sklearn, matplotlib, seaborn, regex
# pip install pandas scikit-learn matplotlib seaborn

import re
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import ast

# --------- config ----------
LOG_PATH = "rag_predictions.log"
GOLD_PATH = "gold.csv"     # columns: omop, gold_target
CONF_THRESHOLD = 0.5      # tau, tune on val set
TOP_K = 5                 # if you want top-k evaluation
# ---------------------------

# 1) Parse logs: extract (omop/source, predicted_target, confidence)
def parse_log_line(line):
    # Pattern #1: human-readable RAGSchemaMatcher line
    m = re.search(r"Table match:\s*([^\s]+)\s*->\s*([^\s]+)\s*\(conf:\s*([0-9.]+)\)", line)
    if m:
        src = m.group(1)
        tgt = m.group(2)
        conf = float(m.group(3))
        return {"omop": src, "pred_target": tgt, "confidence": conf}
    # Pattern #2: JSON assistant chunk like {"match": "admissions", "confidence": 0.772, ...}
    m2 = re.search(r"\{.*\"match\"\s*:\s*\"([^\"]+)\".*\"confidence\"\s*:\s*([0-9.]+).*\}", line)
    if m2:
        return {"omop": None, "pred_target": m2.group(1), "confidence": float(m2.group(2))}
    # try to parse pure JSON line if present
    try:
        obj = ast.literal_eval(line.strip())
        if isinstance(obj, dict) and "match" in obj:
            return {"omop": obj.get("omop", None), "pred_target": obj["match"], "confidence": float(obj.get("confidence", 0.0))}
    except Exception:
        pass
    return None

rows = []
with open(LOG_PATH, "r", encoding="utf-8") as f:
    for ln in f:
        parsed = parse_log_line(ln)
        if parsed:
            rows.append(parsed)

pred_df = pd.DataFrame(rows)
# If some predictions miss 'omop' (because the log didn't include it), you must join externally.
print(f"Parsed {len(pred_df)} candidate predictions from log.")

# 2) Load gold data
gold = pd.read_csv(GOLD_PATH, dtype=str)
gold = gold.rename(columns={gold.columns[0]: "omop", gold.columns[1]: "gold_target"}) if list(gold.columns)[:2] != ["omop","gold_target"] else gold
gold = gold[["omop","gold_target"]].drop_duplicates()

# 3) Collapse per omop for Top-1 evaluation: keep highest-confidence candidate per omop
pred_df_sorted = pred_df.sort_values(["omop","confidence"], ascending=[True, False])
top1 = pred_df_sorted.groupby("omop", as_index=False).first()  # highest-confidence per source

# merge with gold - some sources might be missing predictions -> they become NaN
eval_df = gold.merge(top1, on="omop", how="left")

# 4) Determine predicted label (binary): predicted positive if pred_target == gold_target AND confidence >= tau
def pred_binary(row, tau=CONF_THRESHOLD):
    if pd.isna(row["pred_target"]): 
        return 0
    return 1 if (row["pred_target"].strip().lower() == row["gold_target"].strip().lower() and float(row["confidence"]) >= tau) else 0

eval_df["y_true"] = 1   # every row in gold corresponds to a true mapping (positive). BUT if dataset has negative rows, change flow.
# If your gold data contains both label rows (0 and 1) then adapt: use label column instead.
eval_df["y_pred"] = eval_df.apply(pred_binary, axis=1)

# But note: if gold contains only positives and you want to consider negatives across all candidate pairs,
# you should build a different evaluation table: For every candidate pair (source,target) in dataset, use gold label 0/1
# The current eval_df is correct for source-centric Top-1 classification.

# 5) Compute confusion matrix and metrics
tn, fp, fn, tp = confusion_matrix(eval_df["y_true"], eval_df["y_pred"], labels=[0,1]).ravel() if False else (0,0,0,0)
# The above will fail because y_true is all 1s in this source-centric layout; instead compute counts simply:
tp = int(((eval_df["y_true"]==1) & (eval_df["y_pred"]==1)).sum())
fn = int(((eval_df["y_true"]==1) & (eval_df["y_pred"]==0)).sum())
# Build a pseudo-FP/TN from candidate-level evaluation if needed; otherwise compute precision/recall directly:
precision = precision_score(eval_df["y_true"], eval_df["y_pred"], zero_division=0)
recall = recall_score(eval_df["y_true"], eval_df["y_pred"], zero_division=0)
f1 = f1_score(eval_df["y_true"], eval_df["y_pred"], zero_division=0)

print("Top-1 Source-centric results (threshold = {:.2f}):".format(CONF_THRESHOLD))
print(f"TP: {tp}, FN: {fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(f"Coverage (sources with a prediction ≥ tau): {eval_df[eval_df['confidence']>=CONF_THRESHOLD]['omop'].nunique() / eval_df['omop'].nunique():.4f}")

# 6) Top-k accuracy: check if gold appears in the top-k predictions
topk = pred_df.sort_values(["omop","confidence"], ascending=[True, False]).groupby("omop").head(TOP_K)
topk_hit = topk.merge(gold, left_on=["omop","pred_target"], right_on=["omop","gold_target"], how="inner")
topk_accuracy = topk_hit["omop"].nunique() / gold["omop"].nunique()
print(f"Top-{TOP_K} Accuracy: {topk_accuracy:.4f} ({topk_hit['omop'].nunique()} hits / {gold['omop'].nunique()})")

# 7) If you want classic confusion matrix over candidate pairs (labelled 0/1), prepare a dataset of all (omop, candidate) pairs and gold label, then compute binary predictions per candidate (confidence >= tau AND predicted==gold)
# 8) Save evaluation output
eval_df.to_csv("evaluation_top1.csv", index=False)
print("Evaluation results saved to evaluation_top1.csv")