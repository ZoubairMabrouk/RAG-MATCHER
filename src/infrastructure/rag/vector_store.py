# src/infrastructure/rag/vector_store.py
"""
FAISS-based vector store for RAG system.

KEY FIXES:
- Auto Flat index for small datasets (< MIN_DOCS_FOR_IVF).
- Safe training: never train IVF_PQ with nx < nlist; fallback to Flat.
- Optional dynamic nlist (clusters) based on dataset size.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss

from src.domain.entities.rag_schema import KnowledgeBaseDocument

logger = logging.getLogger(__name__)


class RAGVectorStore:
    MIN_DOCS_FOR_IVF = 100  # KEY FIX: threshold to use IVF_PQ

    def __init__(
        self,
        dimension: int,
        index_type: str = "auto",        # "auto" | "Flat" | "IVF_PQ"
        n_clusters: int = 100,           # default IVF nlist
        n_subquantizers: int = 8,
        bits_per_subquantizer: int = 8,
        dynamic_nlist: bool = True       # KEY FIX: allow dynamic nlist
    ):
        self._dimension = dimension
        self._index_type_param = index_type
        self._n_clusters = n_clusters
        self._n_subquantizers = n_subquantizers
        self._bits_per_subquantizer = bits_per_subquantizer
        self._dynamic_nlist = dynamic_nlist

        self._index: Optional[faiss.Index] = None
        self._documents: List[KnowledgeBaseDocument] = []
        self._embeddings: Optional[np.ndarray] = None
        self._actual_index_type: Optional[str] = None

        logger.info(f"[RAGVectorStore] Initialized (dim={dimension}, requested_type={index_type})")

    def add_documents(
        self,
        documents: List[KnowledgeBaseDocument],
        embeddings: np.ndarray
    ) -> None:
        if not documents:
            logger.warning("[RAGVectorStore] No documents to add")
            return
        if embeddings.shape[0] != len(documents):
            raise ValueError(f"Mismatch: {len(documents)} docs vs {embeddings.shape[0]} embeddings")
        if embeddings.shape[1] != self._dimension:
            raise ValueError(f"Embedding dim mismatch: expected {self._dimension}, got {embeddings.shape[1]}")

        self._documents.extend(documents)
        self._embeddings = embeddings if self._embeddings is None else np.vstack([self._embeddings, embeddings])

        total_docs = len(self._documents)
        index_type = self._determine_index_type(total_docs)

        if self._index is None or self._actual_index_type != index_type:
            self._create_index(index_type)
            self._actual_index_type = index_type

        self._add_to_index(embeddings)
        logger.info(f"[RAGVectorStore] Total docs={total_docs}, index_type={self._actual_index_type}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[KnowledgeBaseDocument, float]]:
        if self._index is None or len(self._documents) == 0:
            logger.warning("[RAGVectorStore] No documents in store")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(top_k, len(self._documents))
        distances, indices = self._index.search(query_embedding.astype("float32"), k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self._documents):
                doc = self._documents[idx]
                if filters and not self._matches_filters(doc, filters):
                    continue
                # cosine-friendly if embeddings are normalized: exp(-L2)
                similarity = float(np.exp(-distance))
                results.append((doc, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # -------------------- Private --------------------

    def _determine_index_type(self, num_docs: int) -> str:
        """KEY FIX: never choose IVF_PQ when dataset is too small."""
        if self._index_type_param == "Flat":
            return "Flat"
        if self._index_type_param == "IVF_PQ":
            if num_docs < self.MIN_DOCS_FOR_IVF:
                logger.warning(
                    f"[RAGVectorStore] IVF_PQ requested but only {num_docs} docs "
                    f"(min {self.MIN_DOCS_FOR_IVF}). Using Flat."
                )
                return "Flat"
            return "IVF_PQ"

        # auto
        if num_docs < self.MIN_DOCS_FOR_IVF:
            logger.info(f"[RAGVectorStore] Auto -> Flat (num_docs={num_docs})")
            return "Flat"
        logger.info(f"[RAGVectorStore] Auto -> IVF_PQ (num_docs={num_docs})")
        return "IVF_PQ"

    def _create_index(self, index_type: str) -> None:
        if index_type == "Flat":
            self._index = faiss.IndexFlatL2(self._dimension)
            logger.info(f"[RAGVectorStore] Created Flat index (dim={self._dimension})")
            return

        # IVF_PQ
        # KEY FIX: optionally reduce nlist based on dataset size
        if self._dynamic_nlist and self._embeddings is not None:
            num_docs = self._embeddings.shape[0]
            # simple heuristic: nlist ≈ sqrt(N) or N/4 capped
            nlist = max(8, min(self._n_clusters, int(max(8, np.sqrt(num_docs)))))
        else:
            nlist = self._n_clusters

        quantizer = faiss.IndexFlatL2(self._dimension)
        self._index = faiss.IndexIVFPQ(
            quantizer,
            self._dimension,
            nlist,
            self._n_subquantizers,
            self._bits_per_subquantizer,
        )
        logger.info(f"[RAGVectorStore] Created IVF_PQ index (dim={self._dimension}, nlist={nlist}, subq={self._n_subquantizers}, bits={self._bits_per_subquantizer})")

    def _add_to_index(self, embeddings: np.ndarray) -> None:
        if isinstance(self._index, faiss.IndexIVFPQ):
            # KEY FIX: never train with nx < nlist
            nlist = self._index.nlist
            nx = self._embeddings.shape[0]
            if not self._index.is_trained:
                if nx < nlist:
                    logger.warning(
                        f"[RAGVectorStore] IVF_PQ needs at least nlist={nlist} training vectors (have {nx}). "
                        f"Falling back to Flat index to avoid FAISS error."
                    )
                    # fallback to Flat transparently
                    self._index = faiss.IndexFlatL2(self._dimension)
                    self._index.add(self._embeddings.astype("float32"))
                    self._actual_index_type = "Flat"
                    return
                self._index.train(self._embeddings.astype("float32"))
                logger.info("[RAGVectorStore] IVF_PQ training complete")

            self._index.add(embeddings.astype("float32"))
            return

        if isinstance(self._index, faiss.IndexFlatL2):
            self._index.add(embeddings.astype("float32"))
            return

        raise ValueError(f"Unknown index type: {type(self._index)}")

    def _matches_filters(self, doc: KnowledgeBaseDocument, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if hasattr(doc, key):
                if getattr(doc, key) != value:
                    return False
            elif key in doc.metadata:
                if doc.metadata[key] != value:
                    return False
            else:
                return False
        return True

    def clear(self) -> None:
        self._index = None
        self._documents = []
        self._embeddings = None
        self._actual_index_type = None
        logger.info("[RAGVectorStore] Store cleared")
