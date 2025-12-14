"""
RAG-based Schema Matcher for Virtual Renaming.

This module provides semantic matching between U-Schema entities/attributes
and existing database schema objects using RAG (Retrieval-Augmented Generation).

Key Features:
- Builds knowledge base from current schema metadata
- Uses embeddings + FAISS for semantic similarity search
- Optional LLM validation for confidence scoring
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

from click import prompt
import numpy as np

from src.domain.entities.schema import SchemaMetadata, Table, Column
from src.domain.entities.rag_schema import KnowledgeBaseDocument
from src.infrastructure.rag.embedding_service import EmbeddingService
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.llm.llm_client import OpenAILLMClient,LLMClient

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
    to existing database schema objects using embeddings and optional LLM validation.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: RAGVectorStore,
        llm_client: Optional[OpenAILLMClient] = None,
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
        self._llm_client = llm_client
        self._table_threshold = table_accept_threshold
        self._column_threshold = column_accept_threshold
        self._top_k = top_k_search
        
        # Knowledge base state
        self._kb_built = False
        self._schema_metadata: Optional[SchemaMetadata] = None
        
        logger.info(f"[RAGSchemaMatcher] Initialized with thresholds: table={table_accept_threshold}, column={column_accept_threshold}")
    
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
    
    def build_kb(self, schema: SchemaMetadata) -> List[KnowledgeBaseDocument]:
        """
        Build knowledge base documents from schema metadata.
        
        Args:
            schema: Current database schema metadata
            
        Returns:
            List of knowledge base documents for tables and columns
        """
        logger.info(f"[RAGSchemaMatcher] Building KB from schema with {len(schema.tables)} tables")
        
        documents = []
        
        for table in schema.tables:
            # Create table-level document
            table_doc = self._create_table_document(table)
            documents.append(table_doc)
            
            # Create column-level documents
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

# Now you can safely call to_text
        texts = [doc.to_text() for doc in documents]# <-- PAS .text ni .content direct
        embeddings = self._embedding_service.embed(texts)
        embeddings_array = np.asarray(embeddings, dtype="float32")

        # Idéalement, le store sait garder les docs
        self._vector_store.add_documents(documents, embeddings_array)

        self._kb_built = True
        logger.info("[RAGSchemaMatcher] KB indexing completed")
        
    def match_table(
        self, 
        entity_name: str, 
        attributes: List[str],
        hints: Optional[List[str]] = None
    ) -> MatchResult:
        """
        Find the best matching table for a U-Schema entity.
        
        Args:
            entity_name: Name of the U-Schema entity
            attributes: List of attribute names in the entity
            hints: Optional hints for better matching
            
        Returns:
            MatchResult with target table name and confidence
        """
        if not self._kb_built:
            logger.warning("[RAGSchemaMatcher] KB not built, returning no match")
            return MatchResult(None, 0.0, "Knowledge base not built", {})
        
        # Build query text for table matching
        query_text = self._build_table_query(entity_name, attributes, hints or [])
        
        # Generate query embedding
        query_embedding = np.array(self._embedding_service.embed([query_text])[0], dtype='float32')
        
        # Search for similar tables
        candidates = self._vector_store.search(
            query_embedding, 
            top_k=self._top_k,
            filters={"kind": "table"}
        )
        
        if not candidates:
            logger.info(f"[RAGSchemaMatcher] No table candidates found for {entity_name}")
        # if not candidates:
        #     logger.info(f"[FORCE LLM] No retrieval candidates for {entity_name}, still invoking LLM validation")
        #     dummy_doc = type("DummyDoc", (), {"table": "N/A"})()
        #     llm_result = self._llm_validate_table(entity_name, attributes, dummy_doc, [])
        #     return MatchResult(
        #         target_name=None,
        #         confidence=llm_result.get("confidence", 0.0),
        #         rationale="Forced LLM validation (no retrieval candidates)",
        #         extra={"method": "llm-only", "retrieval_score": 0.0, "candidates_count": 0}
        #     )

        # Get best candidate
        if candidates:
            best_doc, retrieval_score = candidates[0]
        else:
            best_doc = type("DummyDoc", (), {"table": "N/A"})()
            retrieval_score = 0.0
        
        # Apply LLM validation if available
        if self._llm_client:
            llm_result = self._llm_validate_table(entity_name, attributes, best_doc, candidates)
            llm_match = llm_result.get("target_name")
            llm_conf  = float(llm_result.get("confidence", 0.0))

            # ---- Apply decision rule ----
            if llm_match:
                target_name = llm_match
                final_confidence = llm_conf
                rationale = f"LLM match accepted: {llm_result.get('rationale', '')}"

                return MatchResult(
                    target_name=target_name,
                    confidence=final_confidence,
                    rationale=rationale,
                    extra={
                        "method": "llm",
                        "retrieval_score": retrieval_score,
                        "candidates_count": len(candidates)
                    }
                )

            else:
                # LLM did not produce a strong enough match
                return MatchResult(
                    target_name=None,
                    confidence=llm_conf,
                    rationale=f"LLM rejected (match={llm_match}, conf={llm_conf:.3f})",
                    extra={
                        "method": "llm",
                        "retrieval_score": retrieval_score,
                        "candidates_count": len(candidates)
                    }
                )
    
    def match_column(
        self, 
        table_name: str, 
        attr_name: str, 
        attr_type: str,
        hints: Optional[List[str]] = None
    ) -> MatchResult:
        """
        Find the best matching column for a U-Schema attribute.
        
        Args:
            table_name: Name of the target table
            attr_name: Name of the U-Schema attribute
            attr_type: Data type of the attribute
            hints: Optional hints for better matching
            
        Returns:
            MatchResult with target column name and confidence
        """
        if not self._kb_built:
            logger.warning("[RAGSchemaMatcher] KB not built, returning no match")
            return MatchResult(None, 0.0, "Knowledge base not built", {})
        
        # Build query text for column matching
        query_text = self._build_column_query(table_name, attr_name, attr_type, hints or [])
        
        # Generate query embedding
        query_embedding = np.array(self._embedding_service.embed([query_text])[0], dtype='float32')
        
        # Search for similar columns in the specific table
        candidates = self._vector_store.search(
            query_embedding, 
            top_k=self._top_k,
            filters={"table": table_name, "kind" : "column"}
        )
        
        if not candidates:
            return MatchResult(None, 0.0, f"No column candidates found in table {table_name}", {})
        
        # Get best candidate
        best_doc, retrieval_score = candidates[0]
        
        # Apply LLM validation if available
        if self._llm_client:
            llm_result = self._llm_validate_column(attr_name, attr_type, best_doc, candidates)
            match_name = llm_result.get("match")
            final_confidence = llm_result["confidence"]
            rationale = llm_result["rationale"]
        # Check threshold
        if llm_result:
            # Extract column name from document ID (format: table.column)
            target_name = match_name
            logger.info(f"[RAGSchemaMatcher] Column match: {attr_name} -> {target_name} (conf: {final_confidence:.3f})")
        else:
            target_name = None
            rationale = f"Below threshold: {rationale}"
        
        return MatchResult(
            target_name=target_name,
            confidence=final_confidence,
            rationale=rationale,
            extra={
                "method": "llm" if self._llm_client else "retrieval",
                "retrieval_score": retrieval_score,
                "table": table_name,
                "candidates_count": len(candidates)
            }
        )
    
    # ---- Private methods --------------------------------------------------------
    
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
            id=f"table::{table.name}",     # <-- OBLIGATOIRE
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
            id=f"column::{table.name}.{column.name}",   # <-- OBLIGATOIRE
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
        """Build query text for table matching."""
        parts = [
            f"Entity: {entity_name}",
            f"Attributes: {', '.join(attributes)}"
        ]
        
        if hints:
            parts.append(f"Hints: {', '.join(hints)}")
        
        return ". ".join(parts)
    
    def _build_column_query(self, table_name: str, attr_name: str, attr_type: str, hints: List[str]) -> str:
        """Build query text for column matching."""
        parts = [
            f"Table: {table_name}",
            f"Attribute: {attr_name}",
            f"Type: {attr_type}"
        ]
        
        if hints:
            parts.append(f"Hints: {', '.join(hints)}")
        
        return ". ".join(parts)
    
    
    def _safe_json_extract(text: str) -> dict:
        """Extracts a JSON-like dict from possibly non-JSON LLM output."""
        if not text:
            return {}

        text = text.strip()

        # Remove markdown fences (```json ... ```)
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        text = re.sub(r"```$", "", text).strip()

        # Try to isolate first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_text = match.group(0)
            try:
                return json.loads(json_text)
            except Exception:
                logger.warning(f"[RAGSchemaMatcher] JSON parse failed. Raw block: {json_text[:120]}")
                return {}

        # 🔁 If no JSON found — fallback to heuristic parsing
        # Example: “The best semantic match for the entity "admission" is the table "products".”
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
            "why": text[:250]  # keep original explanation
        }
        
    def _llm_validate_table(
        self, 
        entity_name: str, 
        attributes: List[str], 
        best_candidate: KnowledgeBaseDocument,
        all_candidates: List[Tuple[KnowledgeBaseDocument, float]]
    ) -> Dict[str, Any]:
        """Use LLM to validate table matching decision."""
        if not self._llm_client:
            return {"confidence": 0.0, "rationale": "No LLM client available"}
        
        # Build context
        candidates_text = []
        for doc, score in all_candidates[:10]:  # Top k candidates
            candidates_text.append(f"- {doc.table}: {doc.content} (score: {score:.3f})")
        
        prompt = f"""
You are an advanced database schema alignment engine, specialized in semantic schema matching for healthcare databases within the database MIMIC-III with there 26 tables. 
Your goal is to determine whether a U-Schema entity corresponds semantically 
to one of the candidate database tables retrieved by a RAG system.

===========================
TASK CONTEXT
===========================
You are given:
1. A U-Schema entity (conceptual object) with a name and a list of attributes.
2. A list of candidate database tables produced by a similarity-search retriever.
   Each candidate includes:
   - table name
   - semantic description
   - column names and datatypes
   - retrieval similarity score
3. You must validate the retriever’s ranking and identify the BEST table match.

Your job is to evaluate:
- The conceptual meaning of the entity.
- Whether its attributes logically belong in a table with the given columns.
- Whether the table plays the expected business/domain role.
- Whether the retriever's top suggestion is semantically reasonable.

===========================
ENTITY TO MATCH
===========================
- Name: {entity_name}
- Attributes: {', '.join(attributes)}

===========================
CANDIDATE TABLES (TOP-K)
===========================
The following tables were retrieved as possible matches, sorted by relevance:
{chr(10).join(candidates_text)}

===========================
MATCHING CRITERIA
===========================
When determining the correct table:
1. **Domain/Conceptual Alignment**
   - Compare the entity name to the table name conceptually.
   - Use synonyms, business meaning, typical domain conventions.

2. **Attribute → Column Compatibility**
   - Do the entity attributes logically fit the columns in the table?
   - Are datatypes compatible (id→integer, date→timestamp, name→text)?
   - Do naming variations match? (e.g., "qty" ≈ "quantity", "price" ≈ "unit_price")

3. **Grouping Coherence**
   - Does the table appear to represent the same business object as the entity?
   - Are attributes naturally belonging together in that table?

4. **Structural Indicators**
   - Tables storing entities usually have identifiers (id, code,…)
   - Relationship or junction tables have FK pairs (e.g., order_id, product_id)
   - Avoid matching a conceptual entity to a junction or log table unless appropriate.

5. **Retriever Validation**
   - The retrieval score is *not enough*. Confirm or override using reasoning.

===========================
STRICT OUTPUT FORMAT
===========================
Respond with a JSON object ONLY:

{{
    "match": "<best_table_name_or_null>",
    "confidence": <0.0-1.0>,
    "why": "Short rationale explaining the semantic decision"
}}

- Return `null` when no candidate is appropriate.
- Confidence must reflect reasoning quality, not retrieval score.
- No text outside of the JSON is allowed.
"""

        try:
            #logger.info(f"[LLM DEBUG] Sending prompt for {entity_name}: {prompt}")
            response = self._llm_client._call_llm(prompt)
            if not response or not response.strip():
                logger.warning(f"[LLM DEBUG] Empty response for {entity_name}, retrying after short delay...")
                sleep(5)
                gc.collect()
                response = self._llm_client._call_llm(prompt)
            sleep(5)
            result = RAGSchemaMatcher._safe_json_extract(response)
            # clean = response.strip()

            # # Remove Markdown fences like ```json ... ```
            # if clean.startswith("```"):
            #     clean = clean.strip("`")
            #     if "json" in clean[:10].lower():
            #         clean = clean[clean.lower().find("json") + 4:].strip()
            #     # Remove trailing ```
            #     if "```" in clean:
            #         clean = clean.split("```")[0].strip()

            # # Parse JSON safely
            # result = json.loads(clean)

            # Adapted return
            
            return {
                "confidence": float(result.get("confidence", 0.0)),
                "rationale": result.get("why", "No explanation"),
                "target_name": result.get("match", None)
            }

        except Exception as e:
            logger.warning(f"[RAGSchemaMatcher] LLM validation failed: {e}")
            return {
                "confidence": 0.0,
                "rationale": f"LLM validation failed: {e}",
                "target_name": None
            }

    def _llm_validate_column(
        self, 
        attr_name: str, 
        attr_type: str, 
        best_candidate: KnowledgeBaseDocument,
        all_candidates: List[Tuple[KnowledgeBaseDocument, float]]
    ) -> Dict[str, Any]:
        """Use LLM to validate column matching decision."""
        if not self._llm_client:
            return {"confidence": 0.0, "rationale": "No LLM client available"}
        
        # Build context
        candidates_text = []
        for doc, score in all_candidates[:10]:  # Top 3 candidates
            col_name = doc.id.split('.')[-1] if '.' in doc.id else doc.id
            candidates_text.append(f"- {col_name}: {doc.content} (score: {score:.3f})")
        
        prompt = f"""
You are an advanced database schema alignment engine. 
Your task is to match a single U-Schema attribute to the BEST column among
the candidate columns retrieved by a vector-based semantic search system.

===========================
TASK CONTEXT
===========================
You are given:
1. An attribute from a conceptual U-Schema entity.
2. The attribute name and datatype.
3. A list of candidate database columns retrieved by a RAG system.
   Each candidate includes:
   - the column name
   - the table it belongs to
   - column description (datatype, constraints, PK/FK, etc.)
   - retrieval similarity score

Your goal is to validate the retriever’s ranking and pick the column that 
best matches the semantic meaning of the attribute.

===========================
ATTRIBUTE TO MATCH
===========================
- Name: {attr_name}
- Type: {attr_type}

===========================
CANDIDATE COLUMNS (TOP-K)
===========================
These columns were retrieved as potential matches:
{chr(10).join(candidates_text)}

===========================
MATCHING CRITERIA
===========================
Evaluate each candidate based on:

1. **Semantic Meaning**
   - Compare the attribute name to the column name conceptually.
   - Recognize common abbreviations and synonyms:
     qty ≈ quantity, desc ≈ description, tel ≈ phone_number, dob ≈ date_of_birth, etc.

2. **Datatype Compatibility (Very Important)**
   - String attributes align with VARCHAR/TEXT columns.
   - Numeric attributes align with INT/DECIMAL columns.
   - Boolean attributes match BIT/BOOLEAN.
   - Date attributes match DATE/DATETIME/TIMESTAMP.
   A column with incompatible datatype should receive low confidence.

3. **Business + Domain Context**
   - Determine whether the column fits the likely domain role:
     - Identifiers → *_id, code, reference
     - Monetary values → price, amount, total
     - Quantities → qty, quantity, count
     - Dates → created_at, updated_at, birth_date
     - Status → status, state, flag

4. **Structural Hints**
   - PK/FK columns may be identifiers.
   - NOT NULL often indicates required fields.

5. **Retriever Validation**
   - Use retrieval score as a clue, not a decision.
   - Override it when semantic or type incompatibility is obvious.

===========================
STRICT OUTPUT REQUIREMENTS
===========================
Respond ONLY with a JSON object:

{{
    "match": "<best_column_name_or_null>",
    "confidence": <0.0-1.0>,
    "why": "Short, concise explanation for the decision"
}}

Rules:
- If none of the candidates is appropriate, return null.
- Do NOT invent columns.
- Do NOT include text outside the JSON.
- Confidence must reflect how well the attribute semantically + structurally 
  aligns with the matched column.

"""

        
        try:
            response = self._llm_client._call_llm(prompt)
            if not response or not response.strip():
                logger.warning(f"[LLM DEBUG] Empty response for {attr_name}, retrying after short delay...")
                sleep(5)
                gc.collect()
                response = self._llm_client._call_llm(prompt)
            sleep(5)
            result = RAGSchemaMatcher._safe_json_extract(response)
            
            # clean = response.strip()

            # # Remove Markdown fences like ```json ... ```
            # if clean.startswith("```"):
            #     clean = clean.strip("`")
            #     if "json" in clean[:10].lower():
            #         clean = clean[clean.lower().find("json") + 4:].strip()
            #     # Remove trailing ```
            #     if "```" in clean:
            #         clean = clean.split("```")[0].strip()

            # # Parse JSON safely
            # result = json.loads(clean)

            # Adapted return
            
            return {
                "confidence": float(result.get("confidence", 0.0)),
                "rationale": result.get("why", "No explanation"),
                "match": result.get("match", None)
            }

        except Exception as e:
            logger.warning(f"[RAGSchemaMatcher] LLM validation failed: {e}")
            return {
                "confidence": 0.0,
                "rationale": f"LLM validation failed: {e}",
                "match": None
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
    llm_client: Optional[OpenAILLMClient] = None,
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
