#!/usr/bin/env python3
"""
Dynamic RAG Virtual Rename runner

What it does
------------
1) Loads U-Schema (JSON) from --uschema-file (or stdin).
2) Gets the current relational schema EITHER by:
     a) introspecting it live from your DI container using --db-url
        (or $DATABASE_URL) and --dialect, OR
     b) loading a previously-exported schema snapshot from --schema-file
        (see scripts/export_schema_to_json.py) -- no DB connection needed.
   --schema-file takes priority when both are given.
3) Builds a semantic KB from the current schema and indexes it in a FAISS store.
4) Runs RAG-based matching:
   - entity -> existing table (virtual rename)
   - attribute -> existing column (virtual rename)
5) Computes a migration plan with DiffEngine (using the matcher) and prints SQL
   statements (no physical RENAME operations).
6) Emits a JSON mapping report to stdout (optional file via --out).


Notes
-----
- For tiny schemas (few docs), start with low thresholds (0.25-0.40).
- The script **never** generates physical RENAME statements; it relies on
  virtual mapping when computing the plan.
"""

from enum import Enum
import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.di_container import DIContainer
from src.domain.entities.schema import (
    USchema, USchemaEntity, USchemaAttribute, DataType,
    SchemaMetadata, Table, Column, ForeignKey, Index
)
from src.domain.entities.rules import NamingConvention
from src.domain.entities.evolution import ChangeType
from src.domain.services.diff_engine import DiffEngine
from src.domain.services.migration_builder import MigrationBuilder

from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
from src.infrastructure.llm.llm_client import OpenAILLMClient, LLMClient, AnthropicLLMClient,BaseLLMClient, GeminiLLMClient


# -------------------- logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("test_rag_generation")


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


def load_schema_from_json(schema_file: str) -> SchemaMetadata:
    """
    Load a SchemaMetadata previously exported by scripts/export_schema_to_json.py.
    Reconstructs Table/Column/ForeignKey/Index dataclasses from plain dicts so
    downstream code (RAGSchemaMatcher, DiffEngine, etc.) sees the exact same
    types it would get from a live inspector.introspect_schema() call.
    """
    with open(schema_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def _parse_dt(val: Optional[str]) -> Optional[datetime]:
        if not val:
            return None
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None

    tables = []
    for t in raw.get("tables", []):
        columns = [
            Column(
                name=c["name"],
                data_type=c["data_type"],
                nullable=c.get("nullable", True),
                primary_key=c.get("primary_key", False),
                unique=c.get("unique", False),
                default_value=c.get("default_value"),
                comment=c.get("comment"),
                constraints=c.get("constraints", []) or [],
            )
            for c in t.get("columns", [])
        ]
        foreign_keys = [
            ForeignKey(
                name=fk["name"],
                column=fk["column"],
                referenced_table=fk["referenced_table"],
                referenced_column=fk["referenced_column"],
                on_delete=fk.get("on_delete", "NO ACTION"),
                on_update=fk.get("on_update", "NO ACTION"),
            )
            for fk in t.get("foreign_keys", [])
        ]
        indexes = [
            Index(
                name=idx["name"],
                columns=idx.get("columns", []),
                unique=idx.get("unique", False),
                index_type=idx.get("index_type"),
            )
            for idx in t.get("indexes", [])
        ]
        tables.append(Table(
            name=t["name"],
            schema=t.get("schema", "public"),
            columns=columns,
            primary_keys=t.get("primary_keys", []) or [],
            foreign_keys=foreign_keys,
            indexes=indexes,
            comment=t.get("comment"),
            row_count=t.get("row_count", 0),
            created_at=_parse_dt(t.get("created_at")),
            modified_at=_parse_dt(t.get("modified_at")),
        ))

    return SchemaMetadata(
        tables=tables,
        views=raw.get("views", []) or [],
        materialized_views=raw.get("materialized_views", []) or [],
        database_name=raw.get("database_name", ""),
        version=raw.get("version", ""),
        introspection_timestamp=_parse_dt(raw.get("introspection_timestamp")),
    )


def pretty_changes(changes):
    grouped: Dict[str, List] = {}
    for c in changes:
        grouped.setdefault(c.change_type.value, []).append(c)
    return grouped


def build_matcher(index_type: str, table_thr: float, col_thr: float, top_k: int) -> RAGSchemaMatcher:
    provider = LocalEmbeddingProvider()
    emb = EmbeddingService(provider)
    store = RAGVectorStore(dimension=provider.dimension, index_type=index_type)
    llm_client = BaseLLMClient(model="llama3.1", temperature=0.1)
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

    # 2) Get the current relational schema: --schema-file (offline, no DB
    #    connection needed) takes priority over live introspection.
    db_url = None
    if args.schema_file:
        if not Path(args.schema_file).exists():
            log.error(f"--schema-file given but not found: {args.schema_file}")
            return 2
        log.info(f"Loading schema snapshot from: {args.schema_file} (no DB connection)")
        current_schema: SchemaMetadata = load_schema_from_json(args.schema_file)
    else:
        db_url = args.db_url or os.getenv("DATABASE_URL")
        if not db_url:
            log.error(
                "No schema source available: pass --schema-file for an offline "
                "snapshot, or --db-url / $DATABASE_URL for a live connection."
            )
            return 2

        container = DIContainer()
        container.configure(db_url, args.dialect)
        inspector = container.get_inspector()
        current_schema = inspector.introspect_schema()

    log.info(f"Current schema: {len(current_schema.tables)} table(s)")

    # 3) Build matcher & KB
    matcher = build_matcher(args.index_type, args.table_threshold, args.column_threshold, args.top_k)
    build_kb_and_index(matcher, current_schema, kb_file=args.kb_file)

    # 4) Do semantic mapping (entity->table, attribute->column)
    mapping_report: Dict[str, Any] = {
        "db_url": db_url,
        "schema_file": args.schema_file,
        "dialect": args.dialect,
        "table_threshold": args.table_threshold,
        "column_threshold": args.column_threshold,
        "entities": [],
    }

    # For DiffEngine: inject matcher to do virtual rename logic
    diff = DiffEngine(NamingConvention(), rag_matcher=matcher)
    print(uschema)
    entities = [
        {
            "name": entity.name,
            "attributes": [
                {
                    "name": attr.name,
                    "data_type": getattr(attr.data_type, "value", str(attr.data_type)),
                    "description": attr.description,
                }
                for attr in entity.attributes
            ],
        }
        for entity in uschema.entities
    ]
    # mapping_report = matcher.match_all_entities(entities)
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

    def make_json_serializable(obj):
        """Convertit récursivement les objets Python en valeurs JSON sérialisables."""
        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, dict):
            return {
                key: make_json_serializable(value)
                for key, value in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [
                make_json_serializable(value)
                for value in obj
            ]

        if hasattr(obj, "model_dump"):
            return make_json_serializable(obj.model_dump())

        if hasattr(obj, "__dict__"):
            return make_json_serializable(obj.__dict__)

        return obj
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": {
                "db_url": db_url,
                "schema_file": args.schema_file,
                "dialect": args.dialect,
                "index_type": args.index_type,
                "table_threshold": args.table_threshold,
                "column_threshold": args.column_threshold,
                "top_k": args.top_k,
            },
            "mapping": make_json_serializable(mapping_report["entities"]),
            "plan": make_json_serializable(changes),
            "sql": make_json_serializable(sql_statements),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

        log.info(f"\nSaved report to: {out_path}")

    # Return non-zero if we created new tables that should have been mapped
    # (heuristic: if many entities mapped to None, you may want to adjust thresholds)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Dynamic RAG virtual rename runner")
    p.add_argument("--uschema-file", default="./scripts/uschema.json",
                   help="Path to U-Schema JSON (use '-' to read from stdin)")
    p.add_argument("--schema-file", default="./scripts/schema_snapshot.json",
                   help="Path to a schema snapshot JSON (see scripts/export_schema_to_json.py). "
                        "When given, no DB connection is made at all -- this takes priority over --db-url.")
    p.add_argument("--db-url", default=None,
                   help="Database URL (overrides $DATABASE_URL if provided). Ignored if --schema-file is given.")
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
    p.add_argument("--out", default=None,
                   help="Optional path to write a JSON report")
    p.add_argument("--kb-file", default="./data/rag/knowledge_base_enriched.jsonl",
                   help="Optional external KB file to augment RAG knowledge base")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))