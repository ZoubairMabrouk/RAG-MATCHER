
import gc
import json
import logging
import re
from time import sleep
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from src.domain.entities.schema import SchemaMetadata, Table, Column
from src.domain.entities.rag_schema import KnowledgeBaseDocument
from src.infrastructure.rag.embedding_service import EmbeddingService
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.llm.llm_client import OpenAILLMClient, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    target_name: Optional[str]
    confidence: float
    rationale: str
    extra: Dict[str, Any]


class RAGSchemaMatcher:
    """
    RAG-based semantic matcher for schema objects using batched LLM requests.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: RAGVectorStore,
        llm_client: Optional[LLMClient] = None,
        table_accept_threshold: float = 0.62,
        column_accept_threshold: float = 0.68,
        top_k_search: int = 5
    ):
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_client = llm_client
        self._table_threshold = table_accept_threshold
        self._column_threshold = column_accept_threshold
        self._top_k = top_k_search
        self._kb_built = False
        self._schema_metadata: Optional[SchemaMetadata] = None
        logger.info(f"[RAGSchemaMatcher] Initialized with thresholds: table={table_accept_threshold}, column={column_accept_threshold}")

    # ---------------- KB build and index ----------------

    def build_kb(self, schema: SchemaMetadata) -> List[KnowledgeBaseDocument]:
        documents = []
        for table in schema.tables:
            documents.append(self._create_table_document(table))
            for column in table.columns:
                documents.append(self._create_column_document(table, column))
        logger.info(f"[RAGSchemaMatcher] Created {len(documents)} KB documents")
        return documents

    def index_kb(self, documents: List[KnowledgeBaseDocument]) -> None:
        if not documents:
            logger.warning("[RAGSchemaMatcher] No documents to index")
            return
        logger.info(f"[RAGSchemaMatcher] Indexing {len(documents)} documents")
        texts = [doc.to_text() for doc in documents]
        embeddings = np.asarray(self._embedding_service.embed(texts), dtype="float32")
        self._vector_store.add_documents(documents, embeddings)
        self._kb_built = True
        logger.info("[RAGSchemaMatcher] KB indexing completed")

    # ---------------- Batched table & column matching ----------------

    def match_all_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batched matching: sends all entities/attributes to the LLM in one request.
        Each entity dict should have 'name' and 'attributes' list.
        Returns mapping report [{entity, matched_table, table_confidence, attributes:[{name, target_column, confidence}]}]
        """
        if not self._llm_client:
            raise RuntimeError("LLM client required for batched matching")

        # 1. Build a single batched prompt
        prompt = "You are a database schema expert. Match each entity and its attributes to the best table and columns.\n"
        prompt += "Return a JSON array like [{\"entity\":..., \"matched_table\":..., \"table_confidence\":..., \"attributes\": [{\"name\":..., \"target_column\":..., \"confidence\":...}]}]\n\n"
        for e in entities:
            prompt += f"Entity: {e['name']}\nAttributes: {', '.join(e['attributes'])}\n\n"

        # 2. Call LLM once
        try:
            response_text = self._llm_client._call_llm(prompt)
            if not response_text or not response_text.strip():
                logger.warning("[RAGSchemaMatcher] Empty LLM response, retrying...")
                sleep(2)
                gc.collect()
                response_text = self._llm_client._call_llm(prompt)
            sleep(2)
        except Exception as e:
            logger.warning(f"[RAGSchemaMatcher] LLM call failed: {e}")
            return []

        # 3. Parse JSON safely
        try:
            mapping_report = json.loads(response_text)
        except json.JSONDecodeError:
            mapping_report = self._safe_json_extract(response_text)

        return mapping_report

    # ---------------- Individual table/column fallback ----------------

    def match_table(self, entity_name: str, attributes: List[str]) -> MatchResult:
        """
        Fallback for single-entity match (optional)
        """
        if not self._kb_built:
            return MatchResult(None, 0.0, "KB not built", {})

        query_text = f"Entity: {entity_name}. Attributes: {', '.join(attributes)}"
        query_embedding = np.array(self._embedding_service.embed([query_text])[0], dtype='float32')
        candidates = self._vector_store.search(query_embedding, top_k=self._top_k, filters={"kind": "table"})
        if not candidates:
            return MatchResult(None, 0.0, "No table candidates found", {})

        best_doc, retrieval_score = candidates[0]

        # Optionally use LLM per table (fallback)
        if self._llm_client:
            llm_result = self._llm_validate_table(entity_name, attributes, best_doc, candidates)
            final_confidence = max(retrieval_score, llm_result["confidence"])
            rationale = llm_result["rationale"]
        else:
            final_confidence = retrieval_score
            rationale = f"Retrieval-based match: {best_doc.table} (score: {retrieval_score:.3f})"

        target_name = best_doc.table if final_confidence >= self._table_threshold else None
        return MatchResult(target_name, final_confidence, rationale, {"method": "llm" if self._llm_client else "retrieval"})

    # ---------------- Helpers ----------------

    @staticmethod
    def _create_table_document(table: Table) -> KnowledgeBaseDocument:
        cols_desc = [f"{col.name} ({col.data_type})" for col in table.columns]
        description = f"Table {table.name}. Columns: {', '.join(cols_desc)}."
        return KnowledgeBaseDocument(
            id=f"table::{table.name}",
            table=table.name,
            column="*",
            content=description,
            metadata={"kind": "table", "columns": [c.name for c in table.columns]}
        )

    @staticmethod
    def _create_column_document(table: Table, column: Column) -> KnowledgeBaseDocument:
        description = f"Column {table.name}.{column.name}. Type: {column.data_type}."
        return KnowledgeBaseDocument(
            id=f"column::{table.name}.{column.name}",
            table=table.name,
            column=column.name,
            content=description,
            metadata={"kind": "column", "data_type": column.data_type}
        )

    @staticmethod
    def _safe_json_extract(text: str) -> List[Dict[str, Any]]:
        """Fallback parser for non-strict JSON output from LLM."""
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except Exception:
            logger.warning("[RAGSchemaMatcher] Failed to parse LLM output, returning empty list")
            return []

    def _llm_validate_table(self, entity_name, attributes, best_candidate, all_candidates):
        """Optional single-table validation fallback (not batched)."""
        if not self._llm_client:
            return {"confidence": 0.0, "rationale": "No LLM client"}

        prompt = f"""
        Match entity '{entity_name}' with attributes {attributes} to best table from candidates:
        {', '.join([c[0].table for c in all_candidates])}
        Respond with JSON: {{ "match": "table_name", "confidence": 0.0-1.0, "why": "reason" }}
        """
        try:
            resp = self._llm_client._call_llm(prompt)
            sleep(1)
            return self._safe_json_extract(resp)[0] if isinstance(self._safe_json_extract(resp), list) else {}
        except Exception as e:
            return {"confidence": 0.0, "rationale": f"LLM error: {e}"}


# ---------------- Factory ----------------

def create_rag_schema_matcher(
    embedding_service: EmbeddingService,
    vector_store: RAGVectorStore,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    table_threshold: float = 0.62,
    column_threshold: float = 0.68
) -> RAGSchemaMatcher:
    if use_llm and not llm_client:
        raise ValueError("LLM client required when use_llm=True")
    return RAGSchemaMatcher(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_client=llm_client if use_llm else None,
        table_accept_threshold=table_threshold,
        column_accept_threshold=column_threshold
    )
