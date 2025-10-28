#!/usr/bin/env python3
"""
Dynamic RAG Virtual Rename runner - Enhanced Version

Enhancements:
-------------
1) Builds rich textual descriptions for each entity, including all attributes,
   aggregates, and references for improved RAG matching.
2) Uses LLM fallback for ambiguous matches.
3) Handles small schemas with adjustable thresholds.
4) Outputs detailed mapping report and evolution plan (no physical renames).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.di_container import DIContainer
from src.domain.entities.schema import (
    USchema, USchemaEntity, USchemaAttribute, DataType, SchemaMetadata
)
from src.domain.entities.rules import NamingConvention
from src.domain.entities.evolution import ChangeType
from src.domain.services.diff_engine import DiffEngine
from src.domain.services.migration_builder import MigrationBuilder

from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
from src.infrastructure.llm.llm_client import BaseLLMClient

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
    """Load U-Schema JSON into USchema model."""
    entities = []

    for e in uschema_json.get("uSchemaModel", {}).get("entities", []):
        et = e.get("EntityType", {})
        name = et.get("name", "").lower()  # normalize
        attributes = []

        for variation in et.get("variations", []):
            sv = variation.get("StructuralVariation", {})

            # Process properties
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

            # Process aggregates
            for agg_block in sv.get("aggregates", []):
                agg = agg_block.get("Aggregation")
                if agg:
                    attributes.append(USchemaAttribute(
                        name=f"{agg.get('name','').lower()}->{agg.get('target','').lower()}",
                        data_type=DataType.JSON,
                        is_key=False
                    ))

            # Process references
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

def entity_to_text(entity: USchemaEntity) -> str:
    """
    Converts an entity to a rich textual description
    for RAG matching.
    """
    lines = [f"Entity: {entity.name}"]
    for attr in entity.attributes:
        dtype = attr.data_type.value
        key = " (key)" if attr.is_key else ""
        lines.append(f"- Attribute: {attr.name} [{dtype}]{key}")
    return "\n".join(lines)

def pretty_changes(changes):
    grouped: Dict[str, List] = {}
    for c in changes:
        grouped.setdefault(c.change_type.value, []).append(c)
    return grouped

def build_matcher(index_type: str, table_thr: float, col_thr: float, top_k: int) -> RAGSchemaMatcher:
    provider = LocalEmbeddingProvider()
    emb = EmbeddingService(provider)
    store = RAGVectorStore(dimension=provider.dimension, index_type=index_type)
    llm_client = BaseLLMClient(model="phi3:mini")
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
    """Build KB from current schema and optional external KB."""
    kb_docs = matcher.build_kb(schema)
    if kb_file and Path(kb_file).exists():
        log.info(f"Loading external KB from: {kb_file}")
        with open(kb_file, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                kb_docs.append(doc.get("content", ""))
    matcher.index_kb(kb_docs)
    log.info(f"KB built & indexed: {len(kb_docs)} documents")

# -------------------- main runner --------------------
def run(args) -> int:
    # 1) Load U-Schema JSON
    if args.uschema_file and args.uschema_file != "-":
        with open(args.uschema_file, "r", encoding="utf-8") as f:
            uschema_json = json.load(f)
    else:
        uschema_json = json.load(sys.stdin)

    uschema = load_uschema(uschema_json)
    log.info(f"Loaded U-Schema with {len(uschema.entities)} entities")

    # 2) Configure DI and introspect current schema
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        db_url="postgresql://test:test@localhost:55432/test"
        log.warning("No DB URL provided; using default test URL")

    container = DIContainer()
    container.configure(db_url, args.dialect)
    inspector = container.get_inspector()
    current_schema: SchemaMetadata = inspector.introspect_schema()
    log.info(f"Introspected current schema: {len(current_schema.tables)} table(s)")

    # 3) Build matcher & KB
    matcher = build_matcher(args.index_type, args.table_threshold, args.column_threshold, args.top_k)
    build_kb_and_index(matcher, current_schema, kb_file=args.kb_file)

    # 4) Semantic mapping
    mapping_report: Dict[str, Any] = {
        "db_url": db_url,
        "dialect": args.dialect,
        "table_threshold": args.table_threshold,
        "column_threshold": args.column_threshold,
        "entities": [],
    }

    diff = DiffEngine(NamingConvention(), rag_matcher=matcher)

    for entity in uschema.entities:
        description = entity_to_text(entity)
        attr_names = [a.name for a in entity.attributes]

        # Match table using rich description
        t_res = matcher.match_table(description, attr_names)
        entity_map = {
            "entity": entity.name,
            "matched_table": t_res.target_name,
            "table_confidence": t_res.confidence,
            "table_rationale": t_res.rationale,
            "attributes": [],
        }

        # Match attributes
        if t_res.target_name:
            for a in entity.attributes:
                sql_type = (
                    "INTEGER" if a.data_type == DataType.INTEGER else
                    "DECIMAL(10,2)" if a.data_type == DataType.DECIMAL else
                    "TIMESTAMP" if a.data_type == DataType.TIMESTAMP else
                    "DATE" if a.data_type == DataType.DATE else
                    "BOOLEAN" if a.data_type == DataType.BOOLEAN else
                    "UUID" if a.data_type == DataType.UUID else
                    "VARCHAR(255)"
                )
                c_res = matcher.match_column(
                    t_res.target_name,
                    a.name,
                    sql_type
                )
                entity_map["attributes"].append({
                    "name": a.name,
                    "target_column": c_res.target_name,
                    "confidence": c_res.confidence,
                    "rationale": c_res.rationale,
                })
        else:
            for a in entity.attributes:
                entity_map["attributes"].append({
                    "name": a.name,
                    "target_column": None,
                    "confidence": 0.0,
                    "rationale": "No table match",
                })

        mapping_report["entities"].append(entity_map)

    # 5) Compute evolution plan
    changes = diff.compute_diff(uschema, current_schema)
    grouped = pretty_changes(changes)

    # 6) Build SQL (no renames)
    builder = MigrationBuilder(args.dialect)
    sql_statements = builder.build_migration(changes)

    # 7) Print summary
    log.info("\n=== Semantic Mapping Summary ===")
    for emap in mapping_report["entities"]:
        tgt = emap["matched_table"] or "(new table)"
        log.info(f"- {emap['entity']} -> {tgt} (conf={emap['table_confidence']:.3f})")
        for attr in emap["attributes"]:
            col = attr["target_column"] or "(new column)"
            log.info(f"    · {attr['name']} -> {col} (conf={attr['confidence']:.3f})")

    log.info("\n=== Evolution Plan ===")
    for k, v in grouped.items():
        log.info(f"{k}: {len(v)}")

    log.info("\n=== SQL Statements ===")
    if not sql_statements:
        log.info("(none)")
    else:
        for i, stmt in enumerate(sql_statements, 1):
            log.info(f"{i:02d}. {stmt}")

    # 8) Optional JSON report
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
        log.info(f"Saved report to: {out_path}")

    return 0

def parse_args():
    p = argparse.ArgumentParser(description="Enhanced Dynamic RAG Virtual Rename Runner")
    p.add_argument("--uschema-file", default="./uschema.json", help="Path to U-Schema JSON (use '-' for stdin)")
    p.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="Database URL")
    p.add_argument("--dialect", default="postgresql", choices=["postgresql", "mysql", "sqlite"], help="SQL dialect")
    p.add_argument("--index-type", default="auto", choices=["auto", "Flat", "IVF_PQ"], help="Vector index type")
    p.add_argument("--table-threshold", type=float, default=0.30, help="Table match threshold")
    p.add_argument("--column-threshold", type=float, default=0.25, help="Column match threshold")
    p.add_argument("--top-k", type=int, default=5, help="Top-K candidates")
    p.add_argument("--out", default=None, help="Path to JSON report")
    p.add_argument("--kb-file", default="./data/rag/knowledge_base_enriched.jsonl", help="Optional external KB")
    return p.parse_args()

if __name__ == "__main__":
    sys.exit(run(parse_args()))
