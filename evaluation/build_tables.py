#!/usr/bin/env python3
"""build_reproduction_tables.py — Étape 15. Reads ONLY predictions.jsonl/metrics.json
already on disk under artifacts/reproduction/<split>/<encoder>/<variant>/ and
assembles a comparison table. No metric value is written manually here."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
REPRO_DIR = REPO_ROOT / "artifacts/reproduction"

METRIC_COLS = ["hits@1", "hits@3", "hits@5", "hits@10", "hits@20", "mrr", "mean_rank", "median_rank"]


def collect_rows():
    rows = []
    for metrics_path in REPRO_DIR.rglob("metrics.json"):
        variant_dir = metrics_path.parent
        encoder_dir = variant_dir.parent
        split_dir = encoder_dir.parent
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {
            "split": split_dir.name,
            "encoder": encoder_dir.name,
            "variant": variant_dir.name,
        }
        for col in METRIC_COLS:
            row[col] = metrics.get(col)
        rows.append(row)
    for status_path in REPRO_DIR.rglob("status.json"):
        variant_dir = status_path.parent
        encoder_dir = variant_dir.parent
        split_dir = encoder_dir.parent
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
        row = {"split": split_dir.name, "encoder": encoder_dir.name, "variant": variant_dir.name}
        row.update({col: "BLOCKED" for col in METRIC_COLS})
        row["blocked_reason"] = status.get("reason")
        rows.append(row)
    return rows


def main():
    rows = collect_rows()
    if not rows:
        print("No predictions.jsonl/metrics.json/status.json found under artifacts/reproduction/. "
              "Run scripts/reproduce_memory.py first.")
        return 1

    out_csv = REPRO_DIR / "reproduction_table.csv"
    out_md = REPRO_DIR / "reproduction_table.md"
    out_json = REPRO_DIR / "reproduction_table.json"

    fieldnames = ["split", "encoder", "variant"] + METRIC_COLS + ["blocked_reason"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")

    print(f"Wrote {len(rows)} rows to {out_csv.relative_to(REPO_ROOT)}, {out_md.relative_to(REPO_ROOT)}, "
          f"{out_json.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())