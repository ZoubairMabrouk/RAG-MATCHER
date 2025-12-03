#!/usr/bin/env python3

import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

PRED_FILE = "artifacts/rag_mapping.json"
GOLD_FILE = "datasets/omap/omap_mimic_gold.csv"   # convert XLSX → CSV first

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------
pred = json.load(open(PRED_FILE))
gold = pd.read_csv(GOLD_FILE, dtype=str)

# Normalize
gold["entity"] = gold["entity"].str.lower()
gold["attribute"] = gold["attribute"].str.lower()
gold["gold_table"] = gold["gold_table"].str.lower()
gold["gold_column"] = gold["gold_column"].str.lower()

# Convert prediction JSON → DataFrame
rows = []
for e in pred["mapping"]:
    ent = e["entity"].lower()
    matched_table = (e["matched_table"] or "").lower()

    # TABLE prediction row
    rows.append({
        "entity": ent,
        "attribute": None,
        "pred_table": matched_table,
        "pred_column": None
    })

    # COLUMN prediction rows
    for a in e["attributes"]:
        rows.append({
            "entity": ent,
            "attribute": a["name"].lower(),
            "pred_table": matched_table,
            "pred_column": (a["target_column"] or "").lower()
        })

pred_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# MERGE GOLD + PRED
# ---------------------------------------------------------
# Table-level merge
gold_tables = gold[["entity", "gold_table"]].drop_duplicates()
table_eval = gold_tables.merge(pred_df[pred_df["attribute"].isna()],
                               on="entity", how="left")

# Column-level merge
gold_cols = gold.dropna(subset=["attribute"])
col_eval = gold_cols.merge(pred_df.dropna(subset=["attribute"]),
                           on=["entity", "attribute"], how="left")

# ---------------------------------------------------------
# TABLE-LEVEL METRICS
# ---------------------------------------------------------
table_eval["y_true"] = table_eval["gold_table"]
table_eval["y_pred"] = table_eval["pred_table"]

print("\n===== TABLE-LEVEL METRICS =====\n")
print(classification_report(table_eval["y_true"], table_eval["y_pred"], zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(table_eval["y_true"], table_eval["y_pred"]))

# ---------------------------------------------------------
# COLUMN-LEVEL METRICS
# ---------------------------------------------------------
col_eval["y_true"] = col_eval["gold_column"]
col_eval["y_pred"] = col_eval["pred_column"]

print("\n===== COLUMN-LEVEL METRICS =====\n")
print(classification_report(col_eval["y_true"], col_eval["y_pred"], zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(col_eval["y_true"], col_eval["y_pred"]))

# ---------------------------------------------------------
# SAVE EVAL RESULTS
# ---------------------------------------------------------
table_eval.to_csv("evaluation_table_level.csv", index=False)
col_eval.to_csv("evaluation_column_level.csv", index=False)

print("\nEvaluation saved.")
