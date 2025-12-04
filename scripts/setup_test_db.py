#!/usr/bin/env python3
"""
Dynamic RAG Virtual Rename runner

What it does
------------
1) Loads U-Schema (JSON) from --uschema-file (or stdin).
2) Introspects the current relational schema dynamically from your DI container
   using the provided --db-url (or $DATABASE_URL) and --dialect.
3) Builds a semantic KB from the current schema and indexes it in a FAISS store.
4) Runs RAG-based matching:
   - entity -> existing table (virtual rename)
   - attribute -> existing column (virtual rename)
5) Computes a migration plan with DiffEngine (using the matcher) and prints SQL
   statements (no physical RENAME operations).
6) Emits a JSON mapping report to stdout (optional file via --out).

Usage
-----
python scripts/run_rag_virtual_rename.py \
  --uschema-file path/to/uschema.json \
  --db-url postgresql://user:pass@localhost:5432/db \
  --dialect postgresql \
  --index-type auto \
  --table-threshold 0.35 \
  --column-threshold 0.35 \
  --top-k 5 \
  --out artifacts/rag_mapping.json

Notes
-----
- For tiny schemas (few docs), start with low thresholds (0.25–0.40).
- The script **never** generates physical RENAME statements; it relies on
  virtual mapping when computing the plan.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.di_container import DIContainer
from src.domain.entities.schema import (
    USchema, USchemaEntity, USchemaAttribute, DataType,
    SchemaMetadata
)

from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report)
import pandas as pd
from src.domain.entities.rules import NamingConvention
from src.domain.entities.evolution import ChangeType
from src.domain.services.diff_engine import DiffEngine
from src.domain.services.migration_builder import MigrationBuilder

from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
from src.infrastructure.llm.llm_client import OpenAILLMClient, LLMClient, AnthropicLLMClient,BaseLLMClient


# -------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("run_rag_virtual_rename")


# -------------------- helpers --------------------
_STR2DT = {
    "string": DataType.STRING,
    "integer": DataType.INTEGER,
    "int": DataType.INTEGER,
    "decimal": DataType.DECIMAL,
    "float": DataType.DECIMAL,
    "boolean": DataType.BOOLEAN,
    "bool": DataType.BOOLEAN,
    "timestamp": DataType.TIMESTAMP,
    "datetime": DataType.TIMESTAMP,
    "date": DataType.DATE,
    "json": DataType.JSON,
    "uuid": DataType.UUID,
}

def _to_datatype(val: str) -> DataType:
    if isinstance(val, DataType):
        return val
    key = str(val).strip().lower()
    return _STR2DT.get(key, DataType.STRING)

def load_uschema(uschema_json: Dict[str, Any]) -> USchema:
    entities = []

    for e in uschema_json.get("uSchemaModel", {}).get("entities", []):
        et = e.get("EntityType", {})
        name = et.get("name", "").lower()  # normalize
        attributes = []

        for variation in et.get("variations", []):
            sv = variation.get("StructuralVariation", {})

            for prop_block in sv.get("properties", []):
                attr_list = prop_block.get("Attribute") or []
                if isinstance(attr_list, dict):
                    attr_list = [attr_list]
                for a in attr_list:
                    aname = a.get("name", "").lower()
                    atype = _to_datatype(a.get("type", "string"))
                    iskey = a.get("iskey", False)
                    if aname:
                        attributes.append(USchemaAttribute(
                            name=aname,
                            data_type=atype,
                            is_key=iskey
                        ))

            for agg_block in sv.get("aggregates", []):
                agg = agg_block.get("Aggregation")
                if agg:
                    attributes.append(USchemaAttribute(
                        name=f"{agg.get('name','').lower()}->{agg.get('target','').lower()}",
                        data_type=DataType.JSON,
                        is_key=False
                    ))

            for ref_block in sv.get("references", []):
                ref = ref_block.get("Reference")
                if ref:
                    attributes.append(USchemaAttribute(
                        name=f"{ref.get('name','').lower()}->{ref.get('target','').lower()}",
                        data_type=DataType.JSON,
                        is_key=False
                    ))

        entities.append(USchemaEntity(name=name, attributes=attributes))

    return USchema(entities=entities)



def pretty_changes(changes):
    grouped: Dict[str, List] = {}
    for c in changes:
        grouped.setdefault(c.change_type.value, []).append(c)
    return grouped


def build_matcher(index_type: str, table_thr: float, col_thr: float, top_k: int) -> RAGSchemaMatcher:
    provider = LocalEmbeddingProvider()
    emb = EmbeddingService(provider)
    store = RAGVectorStore(dimension=provider.dimension, index_type=index_type)
    llm_client = BaseLLMClient(model="phi3")
    matcher = RAGSchemaMatcher(
        embedding_service=emb,
        vector_store=store,
        llm_client=llm_client,
        table_accept_threshold=table_thr,
        column_accept_threshold=col_thr,
        top_k_search=top_k,
    )
    return matcher


def build_kb_and_index(matcher: RAGSchemaMatcher, schema: SchemaMetadata, kb_file: str = None):
    kb_docs = matcher.build_kb(schema)
    if kb_file and Path(kb_file).exists():
        log.info(f"Loading external KB from: {kb_file}")
        with open(kb_file, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                #print(doc["content"])
                kb_docs.append(doc["content"])
    matcher.index_kb(kb_docs)
    log.info(f"KB built & indexed: {len(kb_docs)} documents")
    
def load_gold_omap_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str).fillna("")

    # --- Split OMOP entity + attribute ---
    df[["entity", "attribute"]] = df["omop"].str.split("-", n=1, expand=True)

    # --- Split MIMIC table + column ---
    df[["gold_table", "gold_column"]] = df["table"].str.split("-", n=1, expand=True)

    # Keep only needed columns
    df = df[["entity", "attribute", "gold_table", "gold_column", "label"]]

    return df

def evaluate_mapping(mapping_json_path, gold_file_path):
    if not gold_file_path:
        log.info("No gold file provided — skipping evaluation.")
        return

    log.info(f"\n=== Running Evaluation using GOLD dataset: {gold_file_path} ===")

    # ---- Load predictions ----
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    pred_entities = pred["mapping"]

    # ---- Load gold ----
    file_ext = gold_file_path.split(".")[-1].lower()

    if file_ext == "xlsx":
        raw = pd.read_excel(gold_file_path, dtype=str).fillna("")
        gold = load_gold_omap_xlsx(gold_file_path)
    elif file_ext == "csv":
        gold = pd.read_csv(gold_file_path, dtype=str)
    else:
        raise ValueError(f"Unsupported gold file type: {file_ext}")
    
    gold["entity"] = gold["entity"].str.lower()
    gold["attribute"] = gold["attribute"].str.lower()
    gold["gold_table"] = gold["gold_table"].str.lower()
    gold["gold_column"] = gold["gold_column"].str.lower()

    # ---- Build prediction DF ----
    rows = []
    for ent in pred_entities:
        e = ent["entity"].lower()
        matched_table = ent.get("matched_table") or ""

        # Table-level prediction
        rows.append({
            "entity": e,
            "attribute": None,
            "pred_table": matched_table,
            "pred_column": None
        })

        for attr in ent["attributes"]:
            rows.append({
                "entity": e,
                "attribute": attr["name"].lower(),
                "pred_table": matched_table,
                "pred_column": (attr.get("target_column") or "")
            })

    pred_df = pd.DataFrame(rows)

    # ---- Table-level join ----
    gold_tables = gold[["entity", "gold_table"]].drop_duplicates()
    table_eval = gold_tables.merge(
        pred_df[pred_df["attribute"].isna()],
        on="entity", how="left"
    )
    table_eval["y_true"] = table_eval["gold_table"]
    table_eval["y_pred"] = table_eval["pred_table"]

    # ---- Column-level join ----
    gold_cols = gold.dropna(subset=["attribute"])
    col_eval = gold_cols.merge(
        pred_df[pred_df["attribute"].notna()],
        on=["entity", "attribute"], how="left"
    )
    col_eval["y_true"] = col_eval["gold_column"]
    col_eval["y_pred"] = col_eval["pred_column"]

    # ---- Metrics ----
    log.info("\n>>> TABLE-LEVEL METRICS")
    table_report = classification_report(table_eval["y_true"], table_eval["y_pred"], zero_division=0)
    log.info("\n" + table_report)
    table_cm = confusion_matrix(table_eval["y_true"], table_eval["y_pred"])
    log.info(f"Table-level Confusion Matrix:\n{table_cm}")

    log.info("\n>>> COLUMN-LEVEL METRICS")
    col_report = classification_report(col_eval["y_true"], col_eval["y_pred"], zero_division=0)
    log.info("\n" + col_report)
    col_cm = confusion_matrix(col_eval["y_true"], col_eval["y_pred"])
    log.info(f"Column-level Confusion Matrix:\n{col_cm}")

    # ---- Save evaluation artifacts ----
    eval_dir = Path("artifacts/evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    table_eval.to_csv(eval_dir/"table_eval.csv", index=False)
    col_eval.to_csv(eval_dir/"column_eval.csv", index=False)

    with open(eval_dir/"report.json", "w", encoding="utf-8") as f:
        json.dump({
            "table_classification_report": table_report,
            "table_confusion_matrix": table_cm.tolist(),
            "column_classification_report": col_report,
            "column_confusion_matrix": col_cm.tolist(),
        }, f, indent=2)

    log.info(f"\nSaved evaluation results in: {eval_dir}")

def run(args) -> int:
    # 1) Load U-Schema JSON
    if args.uschema_file and args.uschema_file != "-":
        with open(args.uschema_file, "r", encoding="utf-8") as f:
            uschema_json = json.load(f)
    else:
        uschema_json = json.load(sys.stdin)

    uschema = load_uschema(uschema_json)
    # if not uschema.entities:
    #     log.error("U-Schema is empty: no entities found")
    #     return 2

    # 2) Configure DI and introspect current schema dynamically
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        db_url="postgresql://test:test@localhost:55432/test"
        # log.error("Missing --db-url and $DATABASE_URL")
        # return 2

    container = DIContainer()
    container.configure(db_url, args.dialect)
    inspector = container.get_inspector()
    current_schema: SchemaMetadata = inspector.introspect_schema()
    log.info(f"Introspected current schema: {len(current_schema.tables)} table(s)")

    # 3) Build matcher & KB
    matcher = build_matcher(args.index_type, args.table_threshold, args.column_threshold, args.top_k)
    build_kb_and_index(matcher, current_schema, kb_file=args.kb_file)

    # 4) Do semantic mapping (entity->table, attribute->column)
    mapping_report: Dict[str, Any] = {
        "db_url": db_url,
        "dialect": args.dialect,
        "table_threshold": args.table_threshold,
        "column_threshold": args.column_threshold,
        "entities": [],
    }

    # For DiffEngine: inject matcher to do virtual rename logic
    diff = DiffEngine(NamingConvention(), rag_matcher=matcher)
    print(uschema)
    for entity in uschema.entities:
        attr_names = [a.name for a in entity.attributes]
        print("Uschema entities are : ",entity)
        t_res = matcher.match_table(entity.name, attr_names)
        entity_map = {
            "entity": entity.name,
            "matched_table": t_res.target_name,
            "table_confidence": t_res.confidence,
            "table_rationale": t_res.rationale,
            "attributes": [],
        }

        # If a table was found, try each attribute → column
        if t_res.target_name:
            for a in entity.attributes:
                c_res = matcher.match_column(
                    t_res.target_name,
                    a.name,
                    "INTEGER" if a.data_type == DataType.INTEGER else
                    "DECIMAL(10,2)" if a.data_type == DataType.DECIMAL else
                    "TIMESTAMP" if a.data_type == DataType.TIMESTAMP else
                    "DATE" if a.data_type == DataType.DATE else
                    "BOOLEAN" if a.data_type == DataType.BOOLEAN else
                    "UUID" if a.data_type == DataType.UUID else
                    "VARCHAR(255)"
                )
                entity_map["attributes"].append({
                    "name": a.name,
                    "target_column": c_res.target_name,
                    "confidence": c_res.confidence,
                    "rationale": c_res.rationale,
                })

        else:
            # No table mapping — attributes will be treated as new columns on new table
            for a in entity.attributes:
                entity_map["attributes"].append({
                    "name": a.name,
                    "target_column": None,
                    "confidence": 0.0,
                    "rationale": "No table match",
                })

        mapping_report["entities"].append(entity_map)
        

        # 5) Compute evolution plan WITHOUT physical renames
    changes = diff.compute_diff(uschema, current_schema)
    grouped = pretty_changes(changes)

    # 6) Build SQL (no renames generated by builder)
    builder = MigrationBuilder(args.dialect)
    sql_statements = builder.build_migration(changes)

    # 7) Print summary
    log.info("\n=== Semantic Mapping Summary ===")
    for emap in mapping_report["entities"]:
        ent = emap["entity"]
        tgt = emap["matched_table"] or "(new table)"
        log.info(f"- {ent} -> {tgt} (conf={emap['table_confidence']:.3f})")
        for attr in emap["attributes"]:
            col = attr["target_column"] or "(new column)"
            log.info(f"    · {attr['name']} -> {col} (conf={attr['confidence']:.3f})")

    log.info("\n=== Evolution Plan (by type) ===")
    for k, v in grouped.items():
        log.info(f"{k}: {len(v)}")

    log.info("\n=== SQL Statements ===")
    if not sql_statements:
        log.info("(none)")
    else:
        for i, stmt in enumerate(sql_statements, 1):
            log.info(f"{i:02d}. {stmt}")

    # 8) Optional JSON output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "db_url": db_url,
                "dialect": args.dialect,
                "index_type": args.index_type,
                "table_threshold": args.table_threshold,
                "column_threshold": args.column_threshold,
                "top_k": args.top_k,
            },
            "mapping": mapping_report["entities"],
            "plan": [c.model_dump() if hasattr(c, "model_dump") else c.__dict__ for c in changes],
            "sql": sql_statements,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info(f"\nSaved report to: {out_path}")
        # 9) Optional evaluation if gold provided
    evaluate_mapping(str(out_path), "evaluation/data/omap_mimic/omop_mimic_data.xlsx")
    # Return non-zero if we created new tables that should have been mapped
    # (heuristic: if many entities mapped to None, you may want to adjust thresholds)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Dynamic RAG virtual rename runner")
    p.add_argument("--uschema-file", default="./uschema.json",
                   help="Path to U-Schema JSON (use '-' to read from stdin)")
    p.add_argument("--db-url", default=os.getenv("DATABASE_URL"),
                   help="Database URL (overrides $DATABASE_URL if provided)")
    p.add_argument("--dialect", default="postgresql",
                   choices=["postgresql", "mysql", "sqlite"],
                   help="SQL dialect for SQL generation")
    p.add_argument("--index-type", default="auto",
                   choices=["auto", "Flat", "IVF_PQ"],
                   help="Vector index type (use 'auto' or 'Flat' for tiny schemas)")
    p.add_argument("--table-threshold", type=float, default=0.0,
                   help="Accept threshold for table matching")
    p.add_argument("--column-threshold", type=float, default=0.0,
                   help="Accept threshold for column matching")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K candidates to retrieve")
    p.add_argument("--out", default="mapping_report.json",
                   help="Optional path to write a JSON report")
    p.add_argument("--kb-file", default="./data/rag/knowledge_base_enriched.jsonl",
                   help="Optional external KB file to augment RAG knowledge base")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
