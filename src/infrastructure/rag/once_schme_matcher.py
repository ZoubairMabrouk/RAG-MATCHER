
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

    # def match_all_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #     """
    #     Batched matching: sends all entities/attributes to the LLM in one request.
    #     Each entity dict should have 'name' and 'attributes' list.
    #     Returns mapping report [{entity, matched_table, table_confidence, attributes:[{name, target_column, confidence}]}]
    #     """
    #     if not self._llm_client:
    #         raise RuntimeError("LLM client required for batched matching")

    #     # 1. Build a single batched prompt
    #     prompt = "You are a database schema expert. Match each entity and its attributes to the best table and columns.\n"
    #     prompt += "Return a JSON array like [{\"entity\":..., \"matched_table\":..., \"table_confidence\":..., \"attributes\": [{\"name\":..., \"target_column\":..., \"confidence\":...}]}]\n\n"
    #     candidates = []
    #     for e in entities:
    #         prompt += f"Entity: {e['name']}\nAttributes: {', '.join(e['attributes'])}\n\n"
    #         candidates.extend(self._vector_store.search_all([e], top_k=self._top_k))
    #         prompt += f"Top candidate tables: {', '.join([c[0].table for c in candidates])}\n\n"
    #     # 2. Call LLM once
    #     try:
    #         response_text = self._llm_client._call_llm(prompt)
    #         if not response_text or not response_text.strip():
    #             logger.warning("[RAGSchemaMatcher] Empty LLM response, retrying...")
    #             sleep(2)
    #             gc.collect()
    #             response_text = self._llm_client._call_llm(prompt)
    #         sleep(2)
    #     except Exception as e:
    #         logger.warning(f"[RAGSchemaMatcher] LLM call failed: {e}")
    #         return []

    #     # 3. Parse JSON safely
    #     try:
    #         mapping_report = json.loads(response_text)
    #     except json.JSONDecodeError:
    #         mapping_report = self._safe_json_extract(response_text)

    #     return mapping_report
    def match_all_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batched schema matching with retrieval + strict LLM validation.

        Pipeline:
            U-Schema entity
                -> vector retrieval (Top-K)
                -> LLM semantic validation
                -> deterministic confidence calculation elsewhere

        IMPORTANT:
            The LLM does NOT generate confidence scores.
            The LLM can only validate candidates retrieved by the vector store.

        Expected entity format:
            {
                "name": "address",
                "attributes": ["line1", "city", "country"]
            }

        Returns:
            [
                {
                    "entity": "address",
                    "matched_table": None,
                    "decision": "NO_MATCH",
                    "evidence": [...],
                    "matched_attributes": [...]
                }
            ]
        """

        if not self._llm_client:
            raise RuntimeError("LLM client required for batched matching")

        if not entities:
            return []

        # ------------------------------------------------------------------
        # 1. Retrieve candidates independently for each entity
        # ------------------------------------------------------------------

        entity_candidates = []

        for entity in entities:

            entity_name = entity.get("name", "").strip()
            source_attributes = entity.get("attributes", [])

            if not entity_name:
                logger.warning(
                    "[RAGSchemaMatcher] Entity without a name skipped"
                )
                continue

            try:
                retrieved = self._vector_store.search_all(
                    [entity],
                    top_k=self._top_k
                )
            except Exception as exc:
                logger.warning(
                    f"[RAGSchemaMatcher] Candidate retrieval failed "
                    f"for {entity_name}: {exc}"
                )
                retrieved = []

            # IMPORTANT:
            # Do not call the LLM if retrieval found nothing.
            if not retrieved:
                logger.info(
                    f"[RAGSchemaMatcher] No table candidates found for "
                    f"{entity_name}"
                )

                entity_candidates.append({
                    "entity": entity_name,
                    "attributes": source_attributes,
                    "candidates": []
                })

                continue

            # Candidates belong ONLY to this entity.
            candidates = []

            seen_tables = set()

            for result in retrieved:

                try:
                    candidate_schema = result[0]
                    similarity = result[1] if len(result) > 1 else None
                except (IndexError, TypeError):
                    continue

                table_name = getattr(candidate_schema, "table", None)

                if not table_name:
                    continue

                # Avoid duplicate tables.
                if table_name in seen_tables:
                    continue

                seen_tables.add(table_name)

                # Try to recover candidate columns.
                columns = []

                # Depending on your vector-store schema representation,
                # adapt these attribute names if necessary.
                if hasattr(candidate_schema, "columns"):
                    columns = getattr(candidate_schema, "columns") or []

                elif hasattr(candidate_schema, "attributes"):
                    columns = getattr(candidate_schema, "attributes") or []

                # Normalize columns into strings / dictionaries.
                normalized_columns = []

                for column in columns:

                    if isinstance(column, str):
                        normalized_columns.append({
                            "name": column
                        })

                    elif isinstance(column, dict):
                        normalized_columns.append({
                            "name": column.get("name"),
                            "data_type": column.get("data_type"),
                            "description": column.get("description")
                        })

                    else:
                        normalized_columns.append({
                            "name": getattr(column, "name", str(column)),
                            "data_type": str(
                                getattr(column, "data_type", "")
                            ),
                            "description": getattr(
                                column,
                                "description",
                                None
                            )
                        })

                candidates.append({
                    "table": table_name,
                    "columns": normalized_columns,
                    "retrieval_similarity": similarity
                })

            entity_candidates.append({
                "entity": entity_name,
                "attributes": source_attributes,
                "candidates": candidates
            })

        # ------------------------------------------------------------------
        # 2. If nothing was retrieved for ANY entity, do not call the LLM.
        # ------------------------------------------------------------------

        entities_with_candidates = [
            item
            for item in entity_candidates
            if item["candidates"]
        ]

        if not entities_with_candidates:
            logger.info(
                "[RAGSchemaMatcher] No candidates retrieved for any entity. "
                "Skipping LLM validation."
            )

            return [
                {
                    "entity": item["entity"],
                    "matched_table": None,
                    "decision": "NO_MATCH",
                    "evidence": [
                        "No relational table candidate was retrieved."
                    ],
                    "matched_attributes": []
                }
                for item in entity_candidates
            ]

        # ------------------------------------------------------------------
        # 3. Build STRICT validation prompt
        # ------------------------------------------------------------------

        prompt_parts = []

        prompt_parts.append(
            """
    You are a STRICT DATABASE SCHEMA MATCHING VALIDATOR.

    Your task is to validate whether a source U-Schema entity and one
    of the retrieved relational tables represent the SAME underlying
    data concept.

    You are NOT a general-purpose database assistant.

    You MUST follow these rules:

    1. NEVER force a match.
    2. A source entity may have NO matching relational table.
    3. You may ONLY select a table explicitly provided as a candidate.
    4. NEVER invent a table or column.
    5. A semantically related table is NOT necessarily a valid match.
    6. "Contains information about X" does NOT mean "represents X".
    7. Do NOT match entities merely because they occur in the same
    healthcare context.
    8. Do NOT generate confidence scores.
    9. Do NOT generate probabilities.
    10. The application calculates the final score externally.
    11. Prefer NO_MATCH over a speculative match.
    12. Prefer RELATED_BUT_DIFFERENT when the candidate is related
        to the source concept but represents a different entity.

    A valid MATCH requires evidence from multiple signals:

    - conceptual/entity semantic equivalence;
    - supporting attribute correspondence;
    - compatible structural context;
    - compatible data types;
    - lexical similarity only as supporting evidence.

    Entity-level semantic equivalence has priority over contextual
    association.

    IMPORTANT EXAMPLE:

    Source entity:
    address

    Attributes:
    - line1: string
    - city: string
    - country: string

    Candidate table:
    admissions

    Columns:
    - subject_id
    - hadm_id
    - admittime
    - dischtime
    - admission_type
    - admission_location
    - discharge_location
    - insurance
    - language

    Decision:
    NO_MATCH or RELATED_BUT_DIFFERENT.

    Reason:
    admissions represents hospital admission/encounter information.
    It does not represent a postal address. The fact that an admission
    may contain location-related information is not sufficient to make
    it an address entity.

    Another important distinction:

    address != admissions
    patient != admissions
    hospital admission != address

    OUTPUT ONLY VALID JSON.

    Use exactly this structure:

    [
    {
        "entity": "source entity",
        "matched_table": "candidate table or null",
        "decision": "MATCH | RELATED_BUT_DIFFERENT | NO_MATCH",
        "evidence": [
        "short factual reason"
        ],
        "matched_attributes": [
        {
            "source": "source attribute",
            "target": "target column or null",
            "decision": "MATCH | RELATED_BUT_DIFFERENT | NO_MATCH",
            "reason": "short factual reason"
        }
        ]
    }
    ]

    DO NOT output confidence.
    DO NOT output probability.
    DO NOT output a score.
    DO NOT invent candidates.
    DO NOT force a match.
    """
        )

        # ------------------------------------------------------------------
        # 4. Add retrieved candidates to the prompt
        # ------------------------------------------------------------------

        for item in entities_with_candidates:

            prompt_parts.append(
                f"\n\nSOURCE ENTITY\n"
                f"====================\n"
                f"Entity: {item['entity']}\n"
                f"Attributes:\n"
            )

            for attribute in item["attributes"]:
                prompt_parts.append(
                    f"- {attribute}\n"
                )

            prompt_parts.append(
                "\nRETRIEVED CANDIDATE TABLES\n"
                "==========================\n"
            )

            for index, candidate in enumerate(
                item["candidates"],
                start=1
            ):

                prompt_parts.append(
                    f"\nCandidate {index}\n"
                    f"Table: {candidate['table']}\n"
                )

                if candidate["retrieval_similarity"] is not None:
                    prompt_parts.append(
                        f"Retrieval similarity: "
                        f"{candidate['retrieval_similarity']}\n"
                    )

                prompt_parts.append("Columns:\n")

                if not candidate["columns"]:
                    prompt_parts.append(
                        "- No column metadata available\n"
                    )
                else:
                    for column in candidate["columns"]:

                        column_name = column.get("name")

                        if not column_name:
                            continue

                        data_type = column.get("data_type")
                        description = column.get("description")

                        column_text = f"- {column_name}"

                        if data_type:
                            column_text += f" [{data_type}]"

                        if description:
                            column_text += f" - {description}"

                        prompt_parts.append(
                            column_text + "\n"
                        )

        prompt = "".join(prompt_parts)

        # ------------------------------------------------------------------
        # 5. Call LLM once
        # ------------------------------------------------------------------

        try:

            response_text = self._llm_client._call_llm(
                prompt,
                temperature=0.0,
                max_tokens=2048
            )

            if not response_text or not response_text.strip():

                logger.warning(
                    "[RAGSchemaMatcher] Empty LLM response, retrying..."
                )

                sleep(2)
                gc.collect()

                response_text = self._llm_client._call_llm(
                    prompt,
                    temperature=0.0,
                    max_tokens=2048
                )

        except Exception as exc:

            logger.warning(
                f"[RAGSchemaMatcher] LLM validation failed: {exc}"
            )

            # Do not fabricate matches if validation fails.
            return [
                {
                    "entity": item["entity"],
                    "matched_table": None,
                    "decision": "NO_MATCH",
                    "evidence": [
                        "LLM validation failed."
                    ],
                    "matched_attributes": []
                }
                for item in entity_candidates
            ]

        # ------------------------------------------------------------------
        # 6. Parse JSON
        # ------------------------------------------------------------------

        try:
            mapping_report = json.loads(response_text)

        except json.JSONDecodeError:

            mapping_report = self._safe_json_extract(
                response_text
            )

        # ------------------------------------------------------------------
        # 7. Defensive validation of LLM output
        #
        # The LLM is NOT allowed to return a table that wasn't retrieved.
        # ------------------------------------------------------------------

        validated_report = []

        # Build:
        # entity -> allowed candidate tables
        allowed_tables = {
            item["entity"]: {
                candidate["table"]
                for candidate in item["candidates"]
            }
            for item in entity_candidates
        }

        # Handle malformed/non-list responses.
        if not isinstance(mapping_report, list):
            logger.warning(
                "[RAGSchemaMatcher] Invalid LLM response format."
            )
            mapping_report = []

        for result in mapping_report:

            if not isinstance(result, dict):
                continue

            entity_name = result.get("entity")

            if not entity_name:
                continue

            matched_table = result.get("matched_table")
            decision = result.get("decision")

            # --------------------------------------------------------------
            # Never allow hallucinated candidates.
            # --------------------------------------------------------------

            if (
                matched_table is not None
                and matched_table not in allowed_tables.get(
                    entity_name,
                    set()
                )
            ):

                logger.warning(
                    f"[RAGSchemaMatcher] LLM selected non-retrieved "
                    f"table '{matched_table}' for '{entity_name}'. "
                    f"Forcing NO_MATCH."
                )

                matched_table = None
                decision = "NO_MATCH"

                evidence = result.get("evidence", [])

                evidence.append(
                    "LLM selected a table that was not among the "
                    "retrieved candidates."
                )

                result["evidence"] = evidence

            # --------------------------------------------------------------
            # Normalize invalid decisions.
            # --------------------------------------------------------------

            if decision not in {
                "MATCH",
                "RELATED_BUT_DIFFERENT",
                "NO_MATCH"
            }:

                decision = "NO_MATCH"

            # --------------------------------------------------------------
            # Remove any confidence generated by the LLM.
            # --------------------------------------------------------------

            result.pop("confidence", None)
            result.pop("table_confidence", None)

            if "attributes" in result:
                for attribute in result["attributes"]:
                    if isinstance(attribute, dict):
                        attribute.pop("confidence", None)

            result["matched_table"] = (
                matched_table
                if decision == "MATCH"
                else None
            )

            result["decision"] = decision

            validated_report.append(result)

        # ------------------------------------------------------------------
        # 8. Add entities that were not returned by the LLM
        # ------------------------------------------------------------------

        returned_entities = {
            result.get("entity")
            for result in validated_report
        }

        for item in entity_candidates:

            if item["entity"] in returned_entities:
                continue

            if not item["candidates"]:

                validated_report.append({
                    "entity": item["entity"],
                    "matched_table": None,
                    "decision": "NO_MATCH",
                    "evidence": [
                        "No relational table candidate was retrieved."
                    ],
                    "matched_attributes": []
                })

            else:

                validated_report.append({
                    "entity": item["entity"],
                    "matched_table": None,
                    "decision": "NO_MATCH",
                    "evidence": [
                        "No valid match was returned by the LLM."
                    ],
                    "matched_attributes": []
                })

        return validated_report

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
