import faiss
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.domain.repositeries.interfaces import IVectorStore
from src.domain.entities.rag_schema import KnowledgeBaseDocument

logger = logging.getLogger(__name__)


class RAGVectorStore:
    """
    Advanced FAISS-based vector store for RAG system.
    Single Responsibility: Vector storage and retrieval with filtering.
    """
    
    def __init__(self, dimension: int = 384, index_type: str = "IVF_PQ"):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ("Flat", "IVF_PQ", "HNSW")
        """
        self._dimension = dimension
        self._index_type = index_type
        self._documents: List[KnowledgeBaseDocument] = []
        self._embeddings: Optional[np.ndarray] = None
        
        # Initialize index based on type
        if index_type == "Flat":
            self._index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings
        elif index_type == "IVF_PQ":
            # IVF with Product Quantization for large-scale search
            quantizer = faiss.IndexFlatIP(dimension)
            self._index = faiss.IndexIVFPQ(quantizer, dimension, 100, 8, 8)
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World
            self._index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self._is_trained = False
        logger.info(f"Initialized {index_type} vector store with dimension {dimension}")
    
    def add_documents(self, documents: List[KnowledgeBaseDocument], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of knowledge base documents
            embeddings: Corresponding embeddings matrix
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Store documents
        self._documents.extend(documents)
        
        # Store embeddings
        if self._embeddings is None:
            self._embeddings = embeddings.astype('float32')
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings.astype('float32')])
        
        # Train index if needed (for IVF_PQ)
        if self._index_type == "IVF_PQ" and not self._is_trained:
            logger.info("Training IVF_PQ index...")
            self._index.train(self._embeddings)
            self._is_trained = True
        
        # Add to index
        if self._index_type == "IVF_PQ":
            self._index.add_with_ids(self._embeddings, np.arange(len(self._embeddings)))
        else:
            self._index.add(self._embeddings)
        
        logger.info(f"Vector store now contains {len(self._documents)} documents")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 20, 
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[KnowledgeBaseDocument, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not self._documents:
            return []
        
        # Ensure query embedding is normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS
        if self._index_type == "Flat":
            scores, indices = self._index.search(
                query_norm.reshape(1, -1).astype('float32'), 
                min(top_k * 3, len(self._documents))  # Get more for filtering
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # For IVF_PQ, search returns distances (lower is better)
            distances, indices = self._index.search(
                query_norm.reshape(1, -1).astype('float32'), 
                min(top_k * 3, len(self._documents))
            )
            scores = 1.0 / (1.0 + distances[0])  # Convert distance to similarity
            indices = indices[0]
        
        # Filter and rank results
        results = []
        for score, idx in zip(scores, indices):
            if idx >= len(self._documents) or idx < 0:
                continue
            
            document = self._documents[idx]
            
            # Apply metadata filters
            if filters and not self._matches_filters(document, filters):
                continue
            
            results.append((document, float(score)))
            
            if len(results) >= top_k:
                break
        
        # Sort by score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _matches_filters(self, document: KnowledgeBaseDocument, filters: Dict[str, Any]) -> bool:
        """Check if document matches the given filters."""
        for key, value in filters.items():
            if key == "table":
                if document.table != value:
                    return False
            elif key == "data_type":
                if document.metadata.get("data_type") != value:
                    return False
            elif key == "type":
                # Custom type filter for specific document types
                if key in document.metadata and document.metadata[key] != value:
                    return False
            elif key == "has_units":
                has_units = document.metadata.get("units") is not None
                if has_units != value:
                    return False
            elif key == "constraint_type":
                constraint_type = document.metadata.get("constraints", {}).get("type")
                if constraint_type != value:
                    return False
        
        return True
    
    def get_document_by_id(self, doc_id: str) -> Optional[KnowledgeBaseDocument]:
        """Get document by ID."""
        for doc in self._documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_documents_by_table(self, table_name: str) -> List[KnowledgeBaseDocument]:
        """Get all documents for a specific table."""
        return [doc for doc in self._documents if doc.table == table_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_documents": len(self._documents),
            "embedding_dimension": self._dimension,
            "index_type": self._index_type,
            "is_trained": self._is_trained,
            "tables": list(set(doc.table for doc in self._documents)),
            "total_columns": len([doc for doc in self._documents if "." in doc.id])
        }
    
    def save(self, path: str) -> None:
        """Save vector store to disk."""
        logger.info(f"Saving vector store to {path}")
        
        # Save FAISS index
        faiss.write_index(self._index, f"{path}.index")
        
        # Save documents and embeddings
        with open(f"{path}.data", 'wb') as f:
            pickle.dump({
                "documents": self._documents,
                "embeddings": self._embeddings,
                "dimension": self._dimension,
                "index_type": self._index_type,
                "is_trained": self._is_trained
            }, f)
        
        logger.info("Vector store saved successfully")
    
    def load(self, path: str) -> None:
        """Load vector store from disk."""
        logger.info(f"Loading vector store from {path}")
        
        # Load FAISS index
        self._index = faiss.read_index(f"{path}.index")
        
        # Load documents and embeddings
        with open(f"{path}.data", 'rb') as f:
            data = pickle.load(f)
            self._documents = data["documents"]
            self._embeddings = data["embeddings"]
            self._dimension = data["dimension"]
            self._index_type = data["index_type"]
            self._is_trained = data["is_trained"]
        
        logger.info(f"Loaded vector store with {len(self._documents)} documents")
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self._documents.clear()
        self._embeddings = None
        self._is_trained = False
        
        # Recreate index
        if self._index_type == "Flat":
            self._index = faiss.IndexFlatIP(self._dimension)
        elif self._index_type == "IVF_PQ":
            quantizer = faiss.IndexFlatIP(self._dimension)
            self._index = faiss.IndexIVFPQ(quantizer, self._dimension, 100, 8, 8)
        elif self._index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(self._dimension, 32)
        
        logger.info("Vector store cleared")


# Legacy FAISSVectorStore for backward compatibility
class FAISSVectorStore(IVectorStore):
    """
    Legacy FAISS-based vector store implementation.
    """
    
    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        self._index = faiss.IndexFlatL2(dimension)
        self._texts = []
        self._metadata = []
    
    def add_embeddings(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings to the store."""
        self._index.add(embeddings.astype('float32'))
        self._texts.extend(texts)
        self._metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar items."""
        # Search in FAISS
        distances, indices = self._index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k * 2  # Get more for filtering
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self._metadata):
                continue
            
            meta = self._metadata[idx]
            
            # Apply filters
            if filters:
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue
            
            results.append({
                "text": self._texts[idx],
                "metadata": meta,
                "distance": float(dist),
                "score": 1.0 / (1.0 + float(dist))
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete_by_metadata(self, filters: Dict[str, Any]) -> None:
        """Delete items by metadata (requires rebuild)."""
        # FAISS doesn't support deletion, so we rebuild
        keep_indices = [
            i for i, meta in enumerate(self._metadata)
            if not all(meta.get(k) == v for k, v in filters.items())
        ]
        
        # Rebuild index
        new_index = faiss.IndexFlatL2(self._dimension)
        new_texts = []
        new_metadata = []
        
        for idx in keep_indices:
            # Extract vector (this is simplified, actual extraction is complex)
            new_texts.append(self._texts[idx])
            new_metadata.append(self._metadata[idx])
        
        self._index = new_index
        self._texts = new_texts
        self._metadata = new_metadata
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        faiss.write_index(self._index, f"{path}.index")
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump({"texts": self._texts, "metadata": self._metadata}, f)
    
    def load(self, path: str) -> None:
        """Load index from disk."""
        self._index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", 'rb') as f:
            data = pickle.load(f)
            self._texts = data["texts"]
            self._metadata = data["metadata"]
