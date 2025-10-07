"""RAG retrieval service."""

from typing import List, Dict, Any
from src.domain.repositeries.interfaces import IVectorStore
from src.infrastructure.rag.embedding_service import EmbeddingService

class RAGRetriever:
    """
    Retrieves relevant context using RAG.
    Single Responsibility: Context retrieval.
    """
    
    def __init__(self, vector_store: IVectorStore, embedding_service: EmbeddingService):
        self._vector_store = vector_store
        self._embedding_service = embedding_service
    
    def retrieve_relevant_tables(
        self, 
        entity_names: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
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
    
    def retrieve_relevant_columns(
        self, 
        table_name: str, 
        attribute_names: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
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
    
    def retrieve_design_rules(
        self, 
        context: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant design rules."""
        query = f"Design rules for: {context}"
        embedding = self._embedding_service._provider.embed([query])[0]
        
        return self._vector_store.search(
            embedding,
            top_k=top_k,
            filters={"type": "rule"}
        )
