#!/usr/bin/env python3
"""
End-to-end smoke test for the updated RAG schema matcher.

What it checks
--------------
1) Builds a KB from a tiny demo relational schema (products table).
2) Indexes with FAISS using Flat (or auto/IVF_PQ if you ask for it).
3) Runs table+column semantic matching:
   - items  -> products
   - qte    -> quantity
   - ref    -> reference
4) Prints top-k table/column candidates with scores + final decision.
5) Exits with non-zero status if expectations fail.

Usage
-----
python scripts/verify_rag_pipeline.py \
  --index-type auto \
  --table-threshold 0.30 \
  --column-threshold 0.30 \
  --top-k 5

Tip: start with low thresholds (0.25–0.40) on tiny KBs.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Make project src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from src.domain.entities.schema import (
    SchemaMetadata, Table, Column,
    USchema, USchemaEntity, USchemaAttribute, DataType
)
from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("verify_rag_pipeline")


def make_demo_current_schema() -> SchemaMetadata:
    """Small demo schema with a single 'products' table."""
    return SchemaMetadata(tables=[
        Table(
            name="products",
            columns=[
                Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                Column(name="name", data_type="VARCHAR(255)", nullable=False),
                Column(name="price", data_type="DECIMAL(10,2)", nullable=False),
                Column(name="quantity", data_type="INTEGER", nullable=True),
                Column(name="reference", data_type="VARCHAR(255)", nullable=True),
            ]
        )
    ])


def make_demo_uschema() -> USchema:
    """USchema that should map to the products table + 2 columns."""
    return USchema(entities=[
        USchemaEntity(
            name="items",  # <-- expect to map to products (virtual rename)
            attributes=[
                USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="name", data_type=DataType.STRING, required=True),
                USchemaAttribute(name="price", data_type=DataType.DECIMAL, required=True),
                USchemaAttribute(name="qte", data_type=DataType.INTEGER, required=False),   # -> quantity
                USchemaAttribute(name="ref", data_type=DataType.STRING, required=False),    # -> reference
            ],
        )
    ])


def print_candidates(kind: str, candidates, max_rows=5):
    log.info(f"Top-{min(len(candidates), max_rows)} {kind} candidates:")
    for i, (doc, score) in enumerate(candidates[:max_rows], 1):
        if kind == "tables":
            name = doc.table
        else:
            # columns
            name = f"{doc.table}.{doc.column}"
        log.info(f"  {i:02d}. {name:30s} | similarity={score:.3f}")


def run(args) -> int:
    # Silence TF info spam if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # 1) Demo data
    current_schema = make_demo_current_schema()
    uschema = make_demo_uschema()
    items_entity = uschema.entities[0]

    # 2) Embeddings + vector store
    provider = LocalEmbeddingProvider()
    emb_svc = EmbeddingService(provider)
    vs = RAGVectorStore(
        dimension=provider.dimension,
        index_type=args.index_type,          # "auto" recommended for tests
    )

    # 3) Matcher
    matcher = RAGSchemaMatcher(
        embedding_service=emb_svc,
        vector_store=vs,
        llm_client=None,                     # no LLM for this smoke test
        table_accept_threshold=args.table_threshold,
        column_accept_threshold=args.column_threshold,
        top_k_search=args.top_k,
    )

    # 4) Build + index KB
    kb_docs = matcher.build_kb(current_schema)
    matcher.index_kb(kb_docs)
    log.info(f"KB built with {len(kb_docs)} documents")

    # 5) Table search diagnostics (manual)
    table_query = f"Entity: items. Attributes: {', '.join(a.name for a in items_entity.attributes)}"
    q_emb = np.asarray(emb_svc.embed([table_query])[0], dtype="float32")
    table_candidates = vs.search(q_emb, top_k=args.top_k, filters={"kind": "table"})
    print_candidates("tables", table_candidates)

    # 6) Official matcher decision (table)
    table_result = matcher.match_table(items_entity.name, [a.name for a in items_entity.attributes])
    log.info(
        f"Table decision: target={table_result.target_name} "
        f"conf={table_result.confidence:.3f} rationale={table_result.rationale}"
    )

    ok_table = (table_result.target_name == "products")

    # 7) Column diagnostics + decisions (only if table passed or we force test anyway)
    ok_qte = ok_ref = False
    if table_result.target_name:
        # manual diagnostics
        for attr_name, attr_type in [("qte", "INTEGER"), ("ref", "VARCHAR(255)")]:
            col_query = f"Table: {table_result.target_name}. Attribute: {attr_name}. Type: {attr_type}"
            col_q_emb = np.asarray(emb_svc.embed([col_query])[0], dtype="float32")
            col_candidates = vs.search(
                col_q_emb, top_k=args.top_k,
                filters={"table": table_result.target_name, "kind": "column"}
            )
            print_candidates(f"columns for {attr_name}", col_candidates)

        # official decisions
        qte_result = matcher.match_column(table_result.target_name, "qte", "INTEGER")
        ref_result = matcher.match_column(table_result.target_name, "ref", "VARCHAR(255)")

        log.info(
            f"Column decision (qte): target={qte_result.target_name} "
            f"conf={qte_result.confidence:.3f} rationale={qte_result.rationale}"
        )
        log.info(
            f"Column decision (ref): target={ref_result.target_name} "
            f"conf={ref_result.confidence:.3f} rationale={ref_result.rationale}"
        )

        ok_qte = (qte_result.target_name == "quantity")
        ok_ref = (ref_result.target_name == "reference")

    # 8) Summary + exit code
    log.info("=" * 60)
    log.info("RESULT SUMMARY")
    log.info("=" * 60)
    log.info(f"items -> products:     {'OK' if ok_table else 'FAIL'}  (threshold={args.table_threshold})")
    log.info(f"qte   -> quantity:     {'OK' if ok_qte else 'FAIL'}  (threshold={args.column_threshold})")
    log.info(f"ref   -> reference:    {'OK' if ok_ref else 'FAIL'}  (threshold={args.column_threshold})")

    success = ok_table and ok_qte and ok_ref
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify RAG matching pipeline on tiny demo schema.")
    parser.add_argument("--index-type", default="auto", choices=["auto", "Flat", "IVF_PQ"],
                        help="FAISS index type. Use 'auto' or 'Flat' for tiny KBs.")
    parser.add_argument("--table-threshold", type=float, default=0.30,
                        help="Accept threshold for table matching.")
    parser.add_argument("--column-threshold", type=float, default=0.30,
                        help="Accept threshold for column matching.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of candidates to retrieve for diagnostics.")
    args = parser.parse_args()
    sys.exit(run(args))
