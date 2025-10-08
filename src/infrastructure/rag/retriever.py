"""
Advanced RAG retrieval service with filtering and reranking.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.embedding_service import RAGEmbeddingService
from src.domain.entities.rag_schema import (
    RetrievalQuery, KnowledgeBaseDocument, SourceField, FieldType
)

logger = logging.getLogger(__name__)


class AdvancedRAGRetriever:
    """
    Advanced RAG retriever with filtering, reranking, and context-aware retrieval.
    Single Responsibility: Context retrieval with intelligent filtering.
    """
    
    def __init__(self, vector_store: RAGVectorStore, embedding_service: RAGEmbeddingService):
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        
        # Pre-computed filters for common field types
        self._type_filters = {
            FieldType.DATETIME: {"data_type": ["timestamp", "datetime", "date"]},
            FieldType.INTEGER: {"data_type": ["int", "integer", "bigint", "smallint"]},
            FieldType.FLOAT: {"data_type": ["float", "double", "numeric", "decimal"]},
            FieldType.TEXT: {"data_type": ["varchar", "text", "char"]},
            FieldType.BOOLEAN: {"data_type": ["boolean", "bool"]},
            FieldType.CODE: {"constraint_type": ["code", "id"]},
            FieldType.ID: {"constraint_type": ["primary", "foreign"]}
        }
    
    def retrieve_candidates(self, query: RetrievalQuery) -> List[Tuple[KnowledgeBaseDocument, float, float]]:
        """
        Retrieve and rerank candidates for a source field.
        
        Args:
            query: Retrieval query object
            
        Returns:
            List of (document, bi_encoder_score, cross_encoder_score) tuples
        """
        start_time = time.time()
        
        # Step 1: Generate query embedding
        query_embedding = self._embedding_service.embed_query(query)
        
        # Step 2: Apply pre-retrieval filters
        filters = self._build_pre_filters(query)
        
        # Step 3: Initial retrieval with bi-encoder
        initial_results = self._vector_store.search(
            query_embedding,
            top_k=min(query.top_k * 3, 50),  # Get more for reranking
            filters=filters
        )
        
        if not initial_results:
            logger.warning(f"No candidates found for query: {query.source_field.path}")
            return []
        
        # Extract documents and bi-encoder scores
        candidate_docs = [doc for doc, score in initial_results]
        bi_encoder_scores = [score for doc, score in initial_results]
        
        # Step 4: Rerank with cross-encoder
        reranked_results = self._embedding_service.rerank_candidates(query, candidate_docs)
        
        # Step 5: Combine scores and apply post-retrieval filters
        final_results = []
        for i, (doc, cross_score) in enumerate(reranked_results):
            bi_score = bi_encoder_scores[i] if i < len(bi_encoder_scores) else 0.0
            
            # Apply post-retrieval filters
            if self._apply_post_filters(doc, query):
                final_results.append((doc, bi_score, float(cross_score)))
            
            if len(final_results) >= query.top_k:
                break
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Retrieved {len(final_results)} candidates in {processing_time:.2f}ms")
        
        return final_results
    
    def _build_pre_filters(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Build pre-retrieval filters based on query characteristics."""
        filters = {}
        source_field = query.source_field
        
        # Type-based filtering
        if source_field.inferred_type in self._type_filters:
            type_filter = self._type_filters[source_field.inferred_type]
            filters.update(type_filter)
        
        # Unit-based filtering
        if source_field.units:
            filters["has_units"] = True
        
        # Semantic filtering
        if "icd" in source_field.name_tokens or "diagnosis" in source_field.coarse_semantics:
            filters["table"] = "DIAGNOSES_ICD"
        elif "lab" in source_field.name_tokens or "laboratory" in source_field.coarse_semantics:
            filters["table"] = "LABEVENTS"
        elif "chart" in source_field.name_tokens or "vital" in source_field.coarse_semantics:
            filters["table"] = "CHARTEVENTS"
        elif "prescription" in source_field.name_tokens or "drug" in source_field.coarse_semantics:
            filters["table"] = "PRESCRIPTIONS"
        
        # Add custom filters from query
        filters.update(query.filters)
        
        return filters
    
    def _apply_post_filters(self, document: KnowledgeBaseDocument, query: RetrievalQuery) -> bool:
        """Apply post-retrieval filters for fine-grained filtering."""
        source_field = query.source_field
        
        # Constraint-based filtering
        constraints = document.metadata.get("constraints", {})
        
        # Skip foreign keys if no corresponding primary key in source
        if constraints.get("is_foreign_key", False):
            if not self._has_corresponding_primary_key(source_field, document):
                return False
        
        # Unit consistency check
        if source_field.units and document.metadata.get("units"):
            if not self._units_compatible(source_field.units, document.metadata["units"]):
                return False
        
        # Temporal consistency check
        if source_field.inferred_type == FieldType.DATETIME:
            if not self._is_temporal_column(document):
                return False
        
        return True
    
    def _has_corresponding_primary_key(self, source_field: SourceField, document: KnowledgeBaseDocument) -> bool:
        """Check if source has corresponding primary key for foreign key."""
        # Simplified check - in practice, would analyze source schema
        # For now, assume we have patient_id, admission_id, etc. in source
        fk_column = document.column.lower()
        
        if "subject_id" in fk_column or "patient_id" in fk_column:
            return any("patient" in token.lower() or "subject" in token.lower() 
                      for token in source_field.neighbors)
        elif "hadm_id" in fk_column or "admission_id" in fk_column:
            return any("admission" in token.lower() or "encounter" in token.lower() 
                      for token in source_field.neighbors)
        elif "icustay_id" in fk_column:
            return any("icu" in token.lower() or "stay" in token.lower() 
                      for token in source_field.neighbors)
        
        return True  # Default to allowing if unsure
    
    def _units_compatible(self, source_units: str, target_units: str) -> bool:
        """Check if units are compatible."""
        # Normalize units
        source_norm = source_units.lower().strip()
        target_norm = target_units.lower().strip()
        
        # Exact match
        if source_norm == target_norm:
            return True
        
        # Common unit mappings
        unit_mappings = {
            "bpm": ["beats per minute", "beats/min", "hr"],
            "mmHg": ["mm Hg", "mmhg", "torr"],
            "°c": ["celsius", "c", "deg c"],
            "°f": ["fahrenheit", "f", "deg f"],
            "%": ["percent", "pct"],
            "/min": ["per minute", "per min", "/minute"]
        }
        
        for canonical, variants in unit_mappings.items():
            if (source_norm in variants or source_norm == canonical) and \
               (target_norm in variants or target_norm == canonical):
                return True
        
        return False
    
    def _is_temporal_column(self, document: KnowledgeBaseDocument) -> bool:
        """Check if column is temporal."""
        column_name = document.column.lower()
        description = document.metadata.get("description", "").lower()
        
        temporal_indicators = [
            "time", "date", "timestamp", "created", "updated", "modified",
            "admit", "discharge", "birth", "death", "start", "end"
        ]
        
        return any(indicator in column_name or indicator in description 
                  for indicator in temporal_indicators)
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a specific table."""
        documents = self._vector_store.get_documents_by_table(table_name)
        
        if not documents:
            return {}
        
        column_types = {}
        total_columns = len(documents)
        
        for doc in documents:
            data_type = doc.metadata.get("data_type", "unknown")
            column_types[data_type] = column_types.get(data_type, 0) + 1
        
        return {
            "table_name": table_name,
            "total_columns": total_columns,
            "column_types": column_types,
            "has_primary_key": any(doc.metadata.get("constraints", {}).get("is_primary_key") 
                                 for doc in documents),
            "has_foreign_keys": any(doc.metadata.get("constraints", {}).get("is_foreign_key") 
                                  for doc in documents),
            "columns_with_units": len([doc for doc in documents 
                                     if doc.metadata.get("units")])
        }
    
    def search_by_semantics(self, semantic_hints: List[str], top_k: int = 10) -> List[KnowledgeBaseDocument]:
        """Search by semantic hints without specific field context."""
        # Create a dummy query for semantic search
        dummy_field = SourceField(
            path="semantic_search",
            name_tokens=semantic_hints,
            inferred_type=FieldType.TEXT,
            hints=semantic_hints,
            coarse_semantics=semantic_hints
        )
        
        query = RetrievalQuery(source_field=dummy_field, top_k=top_k)
        
        results = self.retrieve_candidates(query)
        return [doc for doc, _, _ in results]


# Legacy RAGRetriever for backward compatibility
class RAGRetriever:
    """
    Legacy RAG retriever for backward compatibility.
    """
    
    def __init__(self, vector_store, embedding_service):
        self._vector_store = vector_store
        self._embedding_service = embedding_service
    
    def retrieve_relevant_tables(self, entity_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve tables relevant to given entity names."""
        results = []
        
        for entity_name in entity_names:
            query = f"Table for entity {entity_name}"
            embedding = self._embedding_service._provider.embed([query])[0]
            
            matches = self._vector_store.search(
                embedding, 
                top_k=top_k,
                filters={"type": "table"}
            )
            results.extend(matches)
        
        # Deduplicate by table name
        seen = set()
        unique_results = []
        for r in results:
            table_name = r["metadata"]["name"]
            if table_name not in seen:
                seen.add(table_name)
                unique_results.append(r)
        
        return unique_results[:top_k]
    
    def retrieve_relevant_columns(self, table_name: str, attribute_names: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve columns relevant to given attributes."""
        results = []
        
        for attr_name in attribute_names:
            query = f"Column {attr_name} in table {table_name}"
            embedding = self._embedding_service._provider.embed([query])[0]
            
            matches = self._vector_store.search(
                embedding,
                top_k=top_k,
                filters={"type": "column", "table": table_name}
            )
            results.extend(matches)
        
        return results[:top_k]
    
    def retrieve_design_rules(self, context: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant design rules."""
        query = f"Design rules for: {context}"
        embedding = self._embedding_service._provider.embed([query])[0]
        
        return self._vector_store.search(
            embedding,
            top_k=top_k,
            filters={"type": "rule"}
        )
