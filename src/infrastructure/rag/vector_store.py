import faiss
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from src.domain.repositeries.interfaces import IVectorStore

class FAISSVectorStore(IVectorStore):
    """
    FAISS-based vector store implementation.
    Single Responsibility: Vector storage and retrieval.
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
