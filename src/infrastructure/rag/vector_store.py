"""
FAISS-based vector store for RAG system.

KEY FIX: Automatically uses Flat index for small datasets (< 100 documents)
to avoid FAISS training error with IVF_PQ index.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss

from src.domain.entities.rag_schema import KnowledgeBaseDocument

logger = logging.getLogger(__name__)


class RAGVectorStore:
    """
    Vector store using FAISS for similarity search.
    
    Automatically selects appropriate index type based on dataset size:
    - Flat index for < 100 documents (no training required)
    - IVF_PQ index for >= 100 documents (better performance)
    """
    
    # Threshold for switching between index types
    MIN_DOCS_FOR_IVF = 100
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "auto",
        n_clusters: int = 100,
        n_subquantizers: int = 8,
        bits_per_subquantizer: int = 8
    ):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: "auto", "Flat", or "IVF_PQ"
                - "auto": Automatically choose based on dataset size
                - "Flat": Always use Flat (exact search, slower but no training)
                - "IVF_PQ": Always use IVF_PQ (fast but requires >= 100 docs)
            n_clusters: Number of IVF clusters (for IVF_PQ)
            n_subquantizers: Number of PQ subquantizers (for IVF_PQ)
            bits_per_subquantizer: Bits per subquantizer (for IVF_PQ)
        """
        self._dimension = dimension
        self._index_type_param = index_type
        self._n_clusters = n_clusters
        self._n_subquantizers = n_subquantizers
        self._bits_per_subquantizer = bits_per_subquantizer
        
        # State
        self._index: Optional[faiss.Index] = None
        self._documents: List[KnowledgeBaseDocument] = []
        self._embeddings: Optional[np.ndarray] = None
        self._actual_index_type: Optional[str] = None
        
        logger.info(f"[RAGVectorStore] Initialized with dimension={dimension}, index_type={index_type}")
    
    def add_documents(
        self,
        documents: List[KnowledgeBaseDocument],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents with their embeddings to the store.
        
        Args:
            documents: List of documents to add
            embeddings: Numpy array of embeddings (N x dimension)
        """
        if len(documents) == 0:
            logger.warning("[RAGVectorStore] No documents to add")
            return
        
        if embeddings.shape[0] != len(documents):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {embeddings.shape[0]} embeddings"
            )
        
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {embeddings.shape[1]}"
            )
        
        logger.info(f"[RAGVectorStore] Adding {len(documents)} documents")
        
        # Store documents and embeddings
        self._documents.extend(documents)
        
        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])
        
        # Determine index type
        total_docs = len(self._documents)
        index_type = self._determine_index_type(total_docs)
        
        # Create or update index
        if self._index is None or self._actual_index_type != index_type:
            self._create_index(index_type)
            self._actual_index_type = index_type
        
        # Add embeddings to index
        self._add_to_index(embeddings)
        
        logger.info(
            f"[RAGVectorStore] Total documents: {total_docs}, "
            f"Index type: {self._actual_index_type}"
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[KnowledgeBaseDocument, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filters (e.g., {"table": "patients", "kind": "column"})
            
        Returns:
            List of (document, score) tuples, sorted by similarity
        """
        if self._index is None or len(self._documents) == 0:
            logger.warning("[RAGVectorStore] No documents in store")
            return []
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(top_k, len(self._documents))
        distances, indices = self._index.search(query_embedding.astype('float32'), k)
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self._documents):  # Valid index
                doc = self._documents[idx]
                
                # Apply filters if specified
                if filters and not self._matches_filters(doc, filters):
                    continue
                
                # Convert distance to similarity score (0-1)
                # For L2 distance, use exponential decay
                similarity = np.exp(-distance)
                
                results.append((doc, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k after filtering
        return results[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_documents": len(self._documents),
            "dimension": self._dimension,
            "index_type": self._actual_index_type,
            "has_index": self._index is not None,
            "embeddings_shape": self._embeddings.shape if self._embeddings is not None else None
        }
    
    # ---- Private Methods --------------------------------------------------------
    
    def _determine_index_type(self, num_docs: int) -> str:
        """
        Determine which index type to use based on dataset size.
        
        Args:
            num_docs: Number of documents
            
        Returns:
            "Flat" or "IVF_PQ"
        """
        if self._index_type_param == "Flat":
            return "Flat"
        elif self._index_type_param == "IVF_PQ":
            if num_docs < self.MIN_DOCS_FOR_IVF:
                logger.warning(
                    f"[RAGVectorStore] IVF_PQ requested but only {num_docs} docs "
                    f"(minimum {self.MIN_DOCS_FOR_IVF}). Using Flat index."
                )
                return "Flat"
            return "IVF_PQ"
        else:  # "auto"
            if num_docs < self.MIN_DOCS_FOR_IVF:
                logger.info(
                    f"[RAGVectorStore] Auto-selecting Flat index for {num_docs} documents"
                )
                return "Flat"
            else:
                logger.info(
                    f"[RAGVectorStore] Auto-selecting IVF_PQ index for {num_docs} documents"
                )
                return "IVF_PQ"
    
    def _create_index(self, index_type: str) -> None:
        """
        Create FAISS index.
        
        Args:
            index_type: "Flat" or "IVF_PQ"
        """
        if index_type == "Flat":
            # Simple flat index (exact search, no training required)
            self._index = faiss.IndexFlatL2(self._dimension)
            logger.info(f"[RAGVectorStore] Created Flat index (dimension={self._dimension})")
            
        elif index_type == "IVF_PQ":
            # IVF-PQ index (fast approximate search, requires training)
            quantizer = faiss.IndexFlatL2(self._dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer,
                self._dimension,
                self._n_clusters,
                self._n_subquantizers,
                self._bits_per_subquantizer
            )
            logger.info(
                f"[RAGVectorStore] Created IVF_PQ index "
                f"(dim={self._dimension}, clusters={self._n_clusters}, "
                f"subq={self._n_subquantizers}, bits={self._bits_per_subquantizer})"
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def _add_to_index(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embeddings to add
        """
        if isinstance(self._index, faiss.IndexIVFPQ):
            # IVF_PQ requires training
            if not self._index.is_trained:
                logger.info("[RAGVectorStore] Training IVF_PQ index...")
                training_data = self._embeddings.astype('float32')
                
                # Verify we have enough data
                if len(training_data) < self._n_clusters:
                    raise RuntimeError(
                        f"IVF_PQ requires at least {self._n_clusters} training points, "
                        f"but only {len(training_data)} available. "
                        f"Use index_type='Flat' or 'auto' for small datasets."
                    )
                
                self._index.train(training_data)
                logger.info("[RAGVectorStore] IVF_PQ training complete")
            
            # Add vectors
            self._index.add(embeddings.astype('float32'))
            
        elif isinstance(self._index, faiss.IndexFlatL2):
            # Flat index - no training needed, just add
            self._index.add(embeddings.astype('float32'))
        
        else:
            raise ValueError(f"Unknown index type: {type(self._index)}")
    
    def _matches_filters(self, doc: KnowledgeBaseDocument, filters: Dict[str, Any]) -> bool:
        """
        Check if document matches filters.
        
        Args:
            doc: Document to check
            filters: Filters to apply
            
        Returns:
            True if document matches all filters
        """
        for key, value in filters.items():
            # Check direct attributes
            if hasattr(doc, key):
                if getattr(doc, key) != value:
                    return False
            # Check metadata
            elif key in doc.metadata:
                if doc.metadata[key] != value:
                    return False
            else:
                # Filter key not found
                return False
        
        return True
    
    def clear(self) -> None:
        """Clear all documents and reset index."""
        self._index = None
        self._documents = []
        self._embeddings = None
        self._actual_index_type = None
        logger.info("[RAGVectorStore] Store cleared")
    
    def save(self, path: str) -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save index
        """
        if self._index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self._index, path)
        logger.info(f"[RAGVectorStore] Index saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load index from disk.
        
        Args:
            path: Path to load index from
        """
        self._index = faiss.read_index(path)
        logger.info(f"[RAGVectorStore] Index loaded from {path}")
    
    def __repr__(self) -> str:
        return (
            f"RAGVectorStore(docs={len(self._documents)}, "
            f"dim={self._dimension}, "
            f"type={self._actual_index_type})"
        )


# Factory function
def create_vector_store(
    dimension: int,
    index_type: str = "auto",
    **kwargs
) -> RAGVectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        dimension: Embedding dimension
        index_type: "auto", "Flat", or "IVF_PQ"
        **kwargs: Additional arguments for RAGVectorStore
        
    Returns:
        Configured RAGVectorStore instance
    """
    return RAGVectorStore(
        dimension=dimension,
        index_type=index_type,
        **kwargs
    )