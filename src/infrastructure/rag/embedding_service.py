from typing import List, Tuple, Dict, Optional
import numpy as np
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.domain.entities.rag_schema import KnowledgeBaseDocument, SourceField, RetrievalQuery

logger = logging.getLogger(__name__)


class RAGEmbeddingService:
    """
    Advanced embedding service for RAG with bi-encoder and cross-encoder.
    Single Responsibility: Embedding generation and similarity scoring.
    """
    
    def __init__(self, bi_encoder_model: str = "all-MiniLM-L6-v2", 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize embedding service with bi-encoder and cross-encoder models.
        
        Args:
            bi_encoder_model: Model for bi-encoder (fast retrieval)
            cross_encoder_model: Model for cross-encoder (accurate reranking)
        """
        logger.info(f"Loading bi-encoder model: {bi_encoder_model}")
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        
        logger.info(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Get embedding dimension
        self.dimension = self.bi_encoder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_documents(self, documents: List[KnowledgeBaseDocument]) -> np.ndarray:
        """
        Generate embeddings for knowledge base documents.
        
        Args:
            documents: List of knowledge base documents
            
        Returns:
            Array of embeddings
        """
        texts = [doc.to_text() for doc in documents]
        
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.bi_encoder.encode(
            texts, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        return embeddings.astype('float32')
    
    def embed_query(self, query: RetrievalQuery) -> np.ndarray:
        """
        Generate embedding for a retrieval query.
        
        Args:
            query: Retrieval query object
            
        Returns:
            Query embedding
        """
        query_text = self._build_query_text(query)
        
        embedding = self.bi_encoder.encode(
            [query_text], 
            normalize_embeddings=True
        )
        
        return embedding[0].astype('float32')
    
    def rerank_candidates(self, query: RetrievalQuery, 
                         candidates: List[KnowledgeBaseDocument]) -> List[Tuple[KnowledgeBaseDocument, float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Original query
            candidates: List of candidate documents
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not candidates:
            return []
        
        query_text = self._build_query_text(query)
        candidate_texts = [doc.to_text() for doc in candidates]
        
        # Create query-candidate pairs
        pairs = [(query_text, candidate_text) for candidate_text in candidate_texts]
        
        # Get reranking scores
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Sort by score (higher is better)
        scored_candidates = list(zip(candidates, rerank_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and candidates.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Matrix of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        # Ensure embeddings are normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidates_norm = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(candidates_norm, query_norm)
        
        return similarities
    
    def _build_query_text(self, query: RetrievalQuery) -> str:
        """Build searchable text from query object."""
        source_field = query.source_field
        
        parts = [
            f"Field: {source_field.path}",
            f"Name tokens: {', '.join(source_field.name_tokens)}",
            f"Type: {source_field.inferred_type}",
        ]
        
        if source_field.format_regex:
            parts.append(f"Format: {source_field.format_regex}")
        
        if source_field.units:
            parts.append(f"Units: {source_field.units}")
        
        if source_field.category_values:
            parts.append(f"Sample values: {', '.join(source_field.category_values[:3])}")
        
        if source_field.hints:
            parts.append(f"Hints: {', '.join(source_field.hints)}")
        
        if source_field.neighbors:
            parts.append(f"Neighbors: {', '.join(source_field.neighbors)}")
        
        if source_field.coarse_semantics:
            parts.append(f"Semantics: {', '.join(source_field.coarse_semantics)}")
        
        return ". ".join(parts)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


# Legacy embedding service for backward compatibility
class EmbeddingService:
    """
    Legacy embedding service for backward compatibility.
    """
    
    def __init__(self, provider):
        self._provider = provider
    
    def embed_schema_objects(self, schema) -> List[Tuple[str, np.ndarray, Dict]]:
        """Generate embeddings for all schema objects."""
        results = []
        
        for table in schema.tables:
            # Embed table
            text = self._table_to_text(table)
            embedding = self._provider.embed([text])[0]
            metadata = {
                "type": "table",
                "name": table.name,
                "schema": table.schema,
                "row_count": table.row_count
            }
            results.append((text, embedding, metadata))
            
            # Embed each column
            for col in table.columns:
                text = self._column_to_text(table.name, col)
                embedding = self._provider.embed([text])[0]
                metadata = {
                    "type": "column",
                    "table": table.name,
                    "name": col.name,
                    "data_type": col.data_type
                }
                results.append((text, embedding, metadata))
        
        return results
    
    def _table_to_text(self, table) -> str:
        """Convert table to searchable text."""
        col_names = ", ".join([col.name for col in table.columns])
        return f"Table: {table.name}. Columns: {col_names}. Comment: {table.comment or 'None'}"
    
    def _column_to_text(self, table_name: str, col) -> str:
        """Convert column to searchable text."""
        return f"Column: {table_name}.{col.name}. Type: {col.data_type}. Comment: {col.comment or 'None'}"

