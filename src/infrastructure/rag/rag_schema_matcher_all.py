"""
RAG-based Schema Matcher for Virtual Renaming.

This module provides semantic matching between U-Schema entities/attributes
and existing database schema objects using RAG (Retrieval-Augmented Generation).

Key Features:
- Builds knowledge base from current schema metadata
- Uses embeddings + FAISS for semantic similarity search
- Single-shot LLM validation: the ENTIRE target schema + the ENTIRE U-Schema
  are sent in ONE prompt, and the LLM returns ONE full mapping report
  (instead of one LLM call per entity/attribute pair).
- Returns MatchResult with target name, confidence, and rationale
- No physical RENAME operations - only virtual aliasing
"""

import gc
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
import os
import re
from time import sleep
from dataclasses import dataclass

import numpy as np

from src.domain.entities.schema import SchemaMetadata, Table, Column
from src.domain.entities.rag_schema import KnowledgeBaseDocument
from src.infrastructure.rag.embedding_service import EmbeddingService
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.llm.llm_client import BaseLLMClient, OpenAILLMClient, LLMClient
from src.infrastructure.llm.llm_service import LLMService
from src.infrastructure.llm.strategies.gemini import GeminiStrategy
from src.infrastructure.llm.factory import LLMFactory
from src.infrastructure.rag.hybrid_reranker import (
    AttributeSpec,
    EntitySpec,
    HybridReranker,
    column_coverage,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """
    Result of a semantic matching operation.

    Attributes:
        target_name: The name of the matched table/column (None if no match)
        confidence: Confidence score (0.0 to 1.0)
        rationale: Human-readable explanation of the match decision
        extra: Additional metadata (source, method, etc.)
    """
    target_name: Optional[str]
    confidence: float
    rationale: str
    extra: Dict[str, Any]


class RAGSchemaMatcher:
    """
    RAG-based semantic matcher for schema objects.

    Single Responsibility: Semantic matching of U-Schema entities/attributes
    to existing database schema objects using embeddings and a SINGLE
    LLM validation call over the whole schema + whole U-Schema.
    """
    from src.infrastructure.llm.factory import LLMFactory

    llm_client = BaseLLMClient(model="llama3.1", temperature=0.1)

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: RAGVectorStore,
        llm_client: Optional[LLMService],
        table_accept_threshold: float = 0.62,
        column_accept_threshold: float = 0.68,
        top_k_search: int = 5
    ):
        """
        Initialize the RAG schema matcher.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
            llm_client: Optional LLM client for validation
            table_accept_threshold: Minimum confidence for table matching
            column_accept_threshold: Minimum confidence for column matching
            top_k_search: Number of candidates to retrieve from vector search
        """
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_client = BaseLLMClient(model="llama3.1", temperature=0.1) if llm_client is None else llm_client
        self._table_threshold = table_accept_threshold
        self._column_threshold = column_accept_threshold
        self._top_k = top_k_search

        # Knowledge base state
        self._kb_built = False
        self._schema_metadata: Optional[SchemaMetadata] = None
        self._hybrid_reranker = HybridReranker(
            accept=table_accept_threshold,
            review=max(0.0, table_accept_threshold - 0.17),
        )

        logger.info(
            f"[RAGSchemaMatcher] Initialized with thresholds: "
            f"table={table_accept_threshold}, column={column_accept_threshold}"
        )

    # ------------------------------------------------------------------
    # KB construction (unchanged)
    # ------------------------------------------------------------------

    def _attribute_to_text(self, attr):
        if isinstance(attr, str):
            return attr

        name = attr.get("name", "")
        data_type = attr.get("data_type", "")
        description = attr.get("description", "")

        return (
            f"Attribute: {name}. "
            f"Data type: {data_type}. "
            f"Description: {description or 'none'}."
        )

    def build_kb(self, schema: SchemaMetadata) -> List[KnowledgeBaseDocument]:
        """
        Build knowledge base documents from schema metadata.

        Args:
            schema: Current database schema metadata

        Returns:
            List of knowledge base documents for tables and columns
        """
        logger.info(f"[RAGSchemaMatcher] Building KB from schema with {len(schema.tables)} tables")
        self._schema_metadata = schema

        documents = []

        for table in schema.tables:
            table_doc = self._create_table_document(table)
            documents.append(table_doc)

            for column in table.columns:
                column_doc = self._create_column_document(table, column)
                documents.append(column_doc)

        logger.info(f"[RAGSchemaMatcher] Created {len(documents)} KB documents")
        return documents

    def index_kb(self, documents: List[KnowledgeBaseDocument]) -> None:
        if not documents:
            logger.warning("[RAGSchemaMatcher] No documents to index")
            return

        logger.info(f"[RAGSchemaMatcher] Indexing {len(documents)} documents")

        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                documents[i] = KnowledgeBaseDocument.from_text(doc)

        texts = [doc.to_text() for doc in documents]
        embeddings = self._embedding_service.embed(texts)
        embeddings_array = np.asarray(embeddings, dtype="float32")

        self._vector_store.add_documents(documents, embeddings_array)

        self._kb_built = True
        logger.info("[RAGSchemaMatcher] KB indexing completed")

    # ------------------------------------------------------------------
    # NEW: single-shot global mapping (one prompt, one LLM call)
    # ------------------------------------------------------------------

    def match_schema_global(
        self,
        entities: List[Dict[str, Any]],
        schema: Optional[SchemaMetadata] = None,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform the ENTIRE entity->table and attribute->column mapping in a
        SINGLE LLM call, instead of one call per entity/attribute pair.

        The prompt embeds:
          - the full target relational schema (all tables, columns, types,
            PK/FK, nullability),
          - the full U-Schema (all entities and all their attributes).

        The LLM is asked to return ONE JSON report covering every entity and
        every attribute at once.

        Args:
            entities: list of dicts like
                {"name": str, "attributes": [{"name","data_type","description"}, ...]}
                (same shape produced by the runner script).
            schema: SchemaMetadata to describe. Defaults to the schema used
                to build the KB (self._schema_metadata).
            max_retries: number of retries on empty/invalid LLM output.

        Returns:
            A mapping report dict:
            {
              "entities": [
                {
                  "entity": str,
                  "matched_table": Optional[str],
                  "table_confidence": float,
                  "table_rationale": str,
                  "attributes": [
                    {"name": str, "target_column": Optional[str],
                     "confidence": float, "rationale": str},
                    ...
                  ]
                },
                ...
              ]
            }
        """
        schema = schema or self._schema_metadata
        if schema is None:
            raise RuntimeError("No schema available: build_kb() must be called first, "
                                "or pass `schema` explicitly.")
        if not self._llm_client:
            raise RuntimeError("LLM client required for global schema matching")

        prompt = self._build_global_prompt(entities, schema)

        response_text = None
        for attempt in range(max_retries + 1):
            try:
                response_text = self._llm_client._call_llm(prompt)
            except Exception as e:
                logger.warning(f"[RAGSchemaMatcher] Global LLM call failed (attempt {attempt}): {e}")
                response_text = None

            if response_text and response_text.strip():
                break

            logger.warning(f"[RAGSchemaMatcher] Empty global LLM response, retrying (attempt {attempt})...")
            sleep(2)
            gc.collect()

        if not response_text or not response_text.strip():
            logger.error("[RAGSchemaMatcher] Global LLM call produced no output after retries")
            return self._empty_global_report(entities, "LLM produced no output")

        parsed = self._safe_json_extract_full(response_text)
        if not parsed or "entities" not in parsed:
            logger.warning("[RAGSchemaMatcher] Could not parse a valid global report, "
                            "returning empty report with raw text preserved")
            report = self._empty_global_report(entities, "Unparseable LLM response")
            report["raw_response"] = response_text[:2000]
            return report

        return self._normalize_global_report(parsed, entities, schema)

    def _build_global_prompt(self, entities: List[Dict[str, Any]], schema: SchemaMetadata) -> str:
        """Build the single prompt describing the whole schema + whole U-Schema."""

        # --- Serialize the full target schema ---
        tables_desc = []
        for table in schema.tables:
            cols = []
            for col in table.columns:
                flags = []
                if getattr(col, "primary_key", False):
                    flags.append("PK")
                if getattr(col, "unique", False):
                    flags.append("UNIQUE")
                if not getattr(col, "nullable", True):
                    flags.append("NOT NULL")
                flag_txt = f" [{', '.join(flags)}]" if flags else ""
                cols.append(f"    - {col.name} ({col.data_type}){flag_txt}")
            fks = [
                f"    - {fk.column} -> {fk.referenced_table}.{fk.referenced_column}"
                for fk in getattr(table, "foreign_keys", []) or []
            ]
            block = [f"Table: {table.name}", "  Columns:"] + cols
            if fks:
                block += ["  Foreign keys:"] + fks
            tables_desc.append("\n".join(block))

        schema_text = "\n\n".join(tables_desc) if tables_desc else "(no tables)"

        # --- Serialize the full U-Schema ---
        entities_desc = []
        for e in entities:
            attr_lines = []
            for a in e.get("attributes", []):
                desc = a.get("description") or "none"
                attr_lines.append(
                    f"    - {a.get('name')} (type: {a.get('data_type')}, description: {desc})"
                )
            entities_desc.append(
                f"Entity: {e.get('name')}\n  Attributes:\n" + "\n".join(attr_lines)
            )
        entities_text = "\n\n".join(entities_desc) if entities_desc else "(no entities)"

        prompt = f"""
You are an advanced database schema alignment engine, specialized in semantic
schema matching for virtual (non-destructive) renaming.

===========================
TASK
===========================
You are given:
1. The COMPLETE target relational database schema (all tables, columns,
   types, primary/foreign keys).
2. The COMPLETE conceptual U-Schema (all entities and all their attributes).

For EVERY entity, decide the best matching table (or null if none fits).
For EVERY attribute of EVERY entity, decide the best matching column within
its matched table (or null if none fits, or if the entity itself has no
table match).

Do this for the WHOLE schema and WHOLE U-Schema at once, in a single
response. Do not ask for more information and do not process entities one
at a time — return one complete report.

===========================
MATCHING CRITERIA
===========================
1. Conceptual/domain alignment between entity name and table name (synonyms,
   business meaning, naming conventions).
2. Attribute -> column compatibility: naming variations (qty ≈ quantity,
   desc ≈ description, dob ≈ date_of_birth, etc.) and datatype compatibility.
3. Structural coherence: identifiers, PK/FK, nullability.
4. An attribute should only be matched to a column that belongs to the table
   already matched to its parent entity — never mix tables.
5. Confidence must reflect reasoning quality, not just superficial similarity.
6. Return null (not a guess) when no candidate is a reasonable semantic fit.

===========================
TARGET DATABASE SCHEMA
===========================
{schema_text}

===========================
U-SCHEMA (SOURCE ENTITIES)
===========================
{entities_text}

===========================
STRICT OUTPUT FORMAT
===========================
Respond with a single JSON object ONLY, no text outside of it, no markdown
fences, covering every entity and every attribute listed above:

{{
  "entities": [
    {{
      "entity": "<entity_name>",
      "matched_table": "<table_name_or_null>",
      "table_confidence": <0.0-1.0>,
      "table_rationale": "<short rationale>",
      "attributes": [
        {{
          "name": "<attribute_name>",
          "target_column": "<column_name_or_null>",
          "confidence": <0.0-1.0>,
          "rationale": "<short rationale>"
        }}
      ]
    }}
  ]
}}
"""
        return prompt

    def _empty_global_report(self, entities: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
        return {
            "entities": [
                {
                    "entity": e.get("name"),
                    "matched_table": None,
                    "table_confidence": 0.0,
                    "table_rationale": reason,
                    "attributes": [
                        {
                            "name": a.get("name"),
                            "target_column": None,
                            "confidence": 0.0,
                            "rationale": reason,
                        }
                        for a in e.get("attributes", [])
                    ],
                }
                for e in entities
            ]
        }

    def _normalize_global_report(
        self,
        parsed: Dict[str, Any],
        entities: List[Dict[str, Any]],
        schema: SchemaMetadata,
    ) -> Dict[str, Any]:
        """
        Validate the LLM's global report against the real schema and the
        thresholds, and fill in any entity/attribute the LLM might have
        omitted so the report always covers the full U-Schema.
        """
        table_names = {t.name for t in schema.tables}
        columns_by_table = {t.name: {c.name for c in t.columns} for t in schema.tables}

        by_entity = {
            item.get("entity"): item
            for item in parsed.get("entities", [])
            if isinstance(item, dict)
        }

        normalized_entities = []
        for e in entities:
            ename = e.get("name")
            item = by_entity.get(ename, {})

            matched_table = item.get("matched_table")
            if matched_table not in table_names:
                if matched_table is not None:
                    logger.warning(
                        f"[RAGSchemaMatcher] LLM proposed unknown table '{matched_table}' "
                        f"for entity '{ename}'; discarding."
                    )
                matched_table = None

            table_conf = float(item.get("table_confidence", 0.0) or 0.0)
            if matched_table and table_conf < self._table_threshold:
                logger.info(
                    f"[RAGSchemaMatcher] Table match '{matched_table}' for '{ename}' "
                    f"below threshold ({table_conf:.3f} < {self._table_threshold}); discarding."
                )
                matched_table = None

            by_attr = {
                a.get("name"): a
                for a in item.get("attributes", [])
                if isinstance(a, dict)
            }

            valid_columns = columns_by_table.get(matched_table, set()) if matched_table else set()

            attributes_out = []
            for a in e.get("attributes", []):
                aname = a.get("name")
                a_item = by_attr.get(aname, {})

                target_column = a_item.get("target_column")
                if not matched_table or target_column not in valid_columns:
                    if target_column is not None:
                        logger.warning(
                            f"[RAGSchemaMatcher] LLM proposed unknown/invalid column "
                            f"'{target_column}' for '{ename}.{aname}'; discarding."
                        )
                    target_column = None

                col_conf = float(a_item.get("confidence", 0.0) or 0.0)
                if target_column and col_conf < self._column_threshold:
                    target_column = None

                attributes_out.append({
                    "name": aname,
                    "target_column": target_column,
                    "confidence": col_conf if target_column else 0.0,
                    "rationale": a_item.get("rationale", "No LLM entry for this attribute"),
                })

            normalized_entities.append({
                "entity": ename,
                "matched_table": matched_table,
                "table_confidence": table_conf if matched_table else 0.0,
                "table_rationale": item.get("table_rationale", "No LLM entry for this entity"),
                "attributes": attributes_out,
            })

        return {"entities": normalized_entities}

    # ------------------------------------------------------------------
    # Legacy per-pair matching (kept for compatibility / fallback use)
    # ------------------------------------------------------------------

    def match_table(
        self,
        entity_name: str,
        attributes: List[str],
        hints: Optional[List[str]] = None
    ) -> MatchResult:
        """Retrieval + hybrid rerank only (no per-call LLM validation)."""
        if not self._kb_built:
            logger.warning("[RAGSchemaMatcher] KB not built, returning no match")
            return MatchResult(None, 0.0, "Knowledge base not built", {})

        query_text = self._build_table_query(entity_name, attributes, hints or [])
        query_embedding = np.array(self._embedding_service.embed([query_text])[0], dtype='float32')
        candidates = self._vector_store.search(
            query_embedding,
            top_k=10,
            filters={"kind": "table"}
        )
        if not candidates:
            return MatchResult(None, 0.0, f"No table candidates found for {entity_name}", {})

        source = EntitySpec(
            name=entity_name,
            embedding=query_embedding,
            attributes=[AttributeSpec(name=attribute) for attribute in attributes],
        )
        table_specs = []
        for doc, _ in candidates:
            table = self._find_table(doc.table)
            table_attributes = [
                AttributeSpec(column.name, str(column.data_type), column.primary_key)
                for column in table.columns
            ] if table else [
                AttributeSpec(column_name) for column_name in doc.metadata.get("columns", [])
            ]
            table_embedding = np.asarray(
                self._embedding_service.embed([doc.content])[0], dtype="float32"
            )
            table_specs.append(EntitySpec(doc.table, table_attributes, table_embedding))

        reranked = self._hybrid_reranker.match(source, table_specs)

        return MatchResult(
            target_name=reranked.target_name,
            confidence=reranked.confidence,
            rationale=reranked.rationale,
            extra={
                "method": "hybrid",
                "decision": reranked.decision.value,
                "margin": reranked.margin,
                "breakdown": reranked.breakdown,
                "top_candidates": reranked.top_candidates,
                "candidates_count": len(candidates),
            },
        )

    def match_column(
        self,
        table_name: str,
        attr_name: str,
        attr_type: str,
        hints: Optional[List[str]] = None
    ) -> MatchResult:
        """Retrieval + hybrid rerank only (no per-call LLM validation)."""
        if not self._kb_built:
            logger.warning("[RAGSchemaMatcher] KB not built, returning no match")
            return MatchResult(None, 0.0, "Knowledge base not built", {})

        query_text = self._build_column_query(table_name, attr_name, attr_type, hints or [])
        query_embedding = np.array(self._embedding_service.embed([query_text])[0], dtype='float32')
        candidates = self._vector_store.search(
            query_embedding,
            top_k=self._top_k,
            filters={"table": table_name, "kind": "column"}
        )
        if not candidates:
            return MatchResult(None, 0.0, f"No column candidates found in table {table_name}", {})

        table = self._find_table(table_name)
        target_attributes = [
            AttributeSpec(column.name, str(column.data_type), column.primary_key)
            for column in table.columns
        ] if table else [
            AttributeSpec(doc.column) for doc, _ in candidates
        ]
        coverage, mapping = column_coverage(
            [AttributeSpec(attr_name, attr_type)], target_attributes
        )
        target_name, column_score = mapping.get(attr_name, (None, 0.0))
        final_confidence = round(column_score, 4)
        target_name = target_name if final_confidence >= self._column_threshold else None

        return MatchResult(
            target_name=target_name,
            confidence=final_confidence,
            rationale=f"Hybrid column score={final_confidence:.3f} for table {table_name}.",
            extra={
                "method": "hybrid",
                "coverage": coverage,
                "table": table_name,
                "candidates_count": len(candidates)
            }
        )

    def _find_table(self, table_name: str) -> Optional[Table]:
        if self._schema_metadata is None:
            return None
        return next(
            (table for table in self._schema_metadata.tables if table.name == table_name),
            None,
        )

    # ---- Private helpers --------------------------------------------------------

    def _create_table_document(self, table: Table) -> KnowledgeBaseDocument:
        cols_desc = []
        for col in table.columns:
            parts = [f"{col.name} ({col.data_type})"]
            if getattr(col, "primary_key", False):
                parts.append("[PK]")
            if getattr(col, "foreign_key", None):
                parts.append("[FK]")
            if not getattr(col, "nullable", True):
                parts.append("[NOT NULL]")
            cols_desc.append(" ".join(parts))

        description = f"Table {table.name}. Columns: {', '.join(cols_desc)}."

        return KnowledgeBaseDocument(
            id=f"table::{table.name}",
            table=table.name,
            column="*",
            content=description,
            metadata={
                "kind": "table",
                "column_count": len(table.columns),
                "columns": [c.name for c in table.columns],
                "primary_keys": [c.name for c in table.columns if getattr(c, "primary_key", False)],
                "foreign_keys": [c.name for c in table.columns if getattr(c, "foreign_key", None)],
            },
        )

    def _create_column_document(self, table: Table, column: Column) -> KnowledgeBaseDocument:
        flags = []
        if getattr(column, "primary_key", False):
            flags.append("PRIMARY KEY")
        if getattr(column, "foreign_key", None):
            flags.append("FOREIGN KEY")
        if not getattr(column, "nullable", True):
            flags.append("NOT NULL")

        constraints = f" [{', '.join(flags)}]" if flags else ""
        description = f"Column {table.name}.{column.name}. Type: {column.data_type}{constraints}."

        return KnowledgeBaseDocument(
            id=f"column::{table.name}.{column.name}",
            table=table.name,
            column=column.name,
            content=description,
            metadata={
                "kind": "column",
                "data_type": column.data_type,
                "is_primary_key": getattr(column, "primary_key", False),
                "is_foreign_key": bool(getattr(column, "foreign_key", None)),
                "is_nullable": getattr(column, "nullable", True),
                "default_value": getattr(column, "default_value", None),
            },
        )

    def _build_table_query(self, entity_name: str, attributes: List[str], hints: List[str]) -> str:
        parts = [
            f"Entity: {entity_name}",
            f"Attributes: {', '.join(attributes)}"
        ]
        if hints:
            parts.append(f"Hints: {', '.join(hints)}")
        return ". ".join(parts)

    def _build_column_query(self, table_name: str, attr_name: str, attr_type: str, hints: List[str]) -> str:
        parts = [
            f"Table: {table_name}",
            f"Attribute: {attr_name}",
            f"Type: {attr_type}"
        ]
        if hints:
            parts.append(f"Hints: {', '.join(hints)}")
        return ". ".join(parts)

    @staticmethod
    def _safe_json_extract_full(text: str) -> Optional[dict]:
        """Extract the full {"entities": [...]} JSON object from LLM output."""
        if not text:
            return None

        text = text.strip()
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        # Fallback: find the largest {...} block and try to parse it.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                logger.warning(f"[RAGSchemaMatcher] Global JSON parse failed. Raw block: {match.group(0)[:200]}")
        return None

    def _safe_json_extract(self, text: str) -> dict:
        """Extracts a JSON-like dict from possibly non-JSON LLM output (legacy, single match)."""
        if not text:
            return {}

        text = text.strip()
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        text = re.sub(r"```$", "", text).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                return json.loads(json_text)
            except Exception:
                logger.warning(f"[RAGSchemaMatcher] JSON parse failed. Raw block: {json_text[:120]}")
                return {}

        m = re.search(r'table\s+"?(\w+)"?', text, re.IGNORECASE)
        match_name = m.group(1) if m else None

        conf = 0.8 if "confidence" not in text.lower() else 0.0
        m_conf = re.search(r'(\d\.\d+)', text)
        if m_conf:
            conf = float(m_conf.group(1))
            conf = min(max(conf, 0.0), 1.0)

        return {
            "match": match_name,
            "confidence": conf,
            "why": text[:250]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get matcher statistics."""
        return {
            "kb_built": self._kb_built,
            "table_threshold": self._table_threshold,
            "column_threshold": self._column_threshold,
            "llm_enabled": self._llm_client is not None,
            "vector_store_stats": self._vector_store.get_statistics()
        }


# Factory function for easy creation
def create_rag_schema_matcher(
    embedding_service: EmbeddingService,
    vector_store: RAGVectorStore,
    use_llm: bool = False,
    llm_client: Optional[LLMService] = None,
    table_threshold: float = 0.62,
    column_threshold: float = 0.68
) -> RAGSchemaMatcher:
    """
    Factory function to create a RAG schema matcher.

    Args:
        embedding_service: Embedding service instance
        vector_store: Vector store instance
        use_llm: Whether to enable LLM validation
        llm_client: LLM client (required if use_llm=True)
        table_threshold: Table matching threshold
        column_threshold: Column matching threshold

    Returns:
        Configured RAGSchemaMatcher instance
    """
    if use_llm and not llm_client:
        raise ValueError("LLM client required when use_llm=True")

    return RAGSchemaMatcher(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_client=llm_client if use_llm else None,
        table_accept_threshold=table_threshold,
        column_accept_threshold=column_threshold
    )