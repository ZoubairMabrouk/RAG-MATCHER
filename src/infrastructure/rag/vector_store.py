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
    # FIX (real root cause of the "clustering ... please provide at least N
    # training points" warnings): FAISS's own rule of thumb for training an
    # IVF_PQ index is ~39 training vectors per centroid. With bits=8 the PQ
    # sub-quantizer needs 2**8=256 centroids -> 256*39 ~= 9984 training points
    # minimum, which is EXACTLY the number the warning was asking for. The
    # previous threshold of 100 was 100x too low: it let "auto" pick IVF_PQ
    # for corpora (e.g. 1282 docs) far too small to ever train it properly,
    # silently degrading retrieval quality with garbage clusters. IVF_PQ only
    # pays off well above ~50k-100k vectors; below that, exact Flat search is
    # both correct AND fast enough. Default raised accordingly.
    MIN_DOCS_FOR_IVF = 20_000

    def __init__(
        self,
        dimension: int,
        index_type: str = "auto",        # "auto" | "Flat" | "IVF_PQ"
        n_clusters: int = 100,           # default IVF nlist
        n_subquantizers: int = 8,
        bits_per_subquantizer: int = 8,
        dynamic_nlist: bool = True,      # KEY FIX: allow dynamic nlist
        metric: str = "ip",              # FIX: "ip" (cosine, for normalized
                                          # embeddings) or "l2". Embeddings
                                          # produced by LocalEmbeddingProvider
                                          # /RAGEmbeddingService are already
                                          # L2-normalized, so inner product
                                          # IS cosine similarity -- no need
                                          # for the previous exp(-L2) hack.
    ):
        self._dimension = dimension
        self._index_type_param = index_type
        self._n_clusters = n_clusters
        self._n_subquantizers = n_subquantizers
        self._bits_per_subquantizer = bits_per_subquantizer
        self._dynamic_nlist = dynamic_nlist

        if metric not in ("ip", "l2"):
            raise ValueError(f"metric must be 'ip' or 'l2', got {metric!r}")
        self._metric = metric
        self._faiss_metric = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2

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

        # FIX: over-fetch before filtering. The previous code searched only
        # `top_k` candidates from FAISS and then applied `filters` -- if the
        # matches for a filter (e.g. kind="table") weren't among the raw
        # top_k nearest neighbors (very likely once column docs vastly
        # outnumber table docs), post-filtering silently returned []. Now we
        # search a wider pool first and filter from that.
        search_k = min(max(top_k * 20, top_k), len(self._documents)) if filters else min(top_k, len(self._documents))
        distances, indices = self._index.search(query_embedding.astype("float32"), search_k)

        results = []
        for idx, raw_score in zip(indices[0], distances[0]):
            if 0 <= idx < len(self._documents):
                doc = self._documents[idx]
                if filters and not self._matches_filters(doc, filters):
                    continue
                # FIX: with metric="ip" and L2-normalized embeddings, FAISS's
                # inner-product score IS the cosine similarity directly --
                # no approximation needed. With metric="l2" (legacy), keep
                # the previous exp(-distance) heuristic for compatibility.
                similarity = float(raw_score) if self._metric == "ip" else float(np.exp(-raw_score))
                results.append((doc, similarity))
                if len(results) >= top_k:
                    break

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
            # FIX: IndexFlatIP instead of IndexFlatL2. Embeddings coming out
            # of LocalEmbeddingProvider/RAGEmbeddingService are already
            # L2-normalized (normalize_embeddings=True), so inner product IS
            # exact cosine similarity -- matches the thesis's stated
            # "cosine similarity top-k retrieval" protocol precisely, instead
            # of approximating it via exp(-L2 distance).
            self._index = (
                faiss.IndexFlatIP(self._dimension) if self._metric == "ip"
                else faiss.IndexFlatL2(self._dimension)
            )
            logger.info(
                "[RAGVectorStore] Created Flat index (dim=%d, metric=%s)",
                self._dimension, self._metric,
            )
            return

        # IVF_PQ (only reached for corpora >= MIN_DOCS_FOR_IVF, i.e. large
        # enough to actually benefit from and train an approximate index)
        if self._dynamic_nlist and self._embeddings is not None:
            num_docs = self._embeddings.shape[0]
            nlist = max(8, min(self._n_clusters, int(max(8, np.sqrt(num_docs)))))
        else:
            nlist = self._n_clusters

        # FIX: quantizer + metric must match the embeddings' geometry (IP for
        # normalized vectors), otherwise the coarse quantizer clusters on the
        # wrong distance and PQ approximates the wrong thing entirely.
        if self._metric == "ip":
            quantizer = faiss.IndexFlatIP(self._dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer, self._dimension, nlist,
                self._n_subquantizers, self._bits_per_subquantizer,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            quantizer = faiss.IndexFlatL2(self._dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer, self._dimension, nlist,
                self._n_subquantizers, self._bits_per_subquantizer,
            )
        logger.info(
            "[RAGVectorStore] Created IVF_PQ index (dim=%d, nlist=%d, subq=%d, bits=%d, metric=%s)",
            self._dimension, nlist, self._n_subquantizers, self._bits_per_subquantizer, self._metric,
        )

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
                    # fallback to Flat transparently (FIX: correct metric)
                    self._index = (
                        faiss.IndexFlatIP(self._dimension) if self._metric == "ip"
                        else faiss.IndexFlatL2(self._dimension)
                    )
                    self._index.add(self._embeddings.astype("float32"))
                    self._actual_index_type = "Flat"
                    return
                self._index.train(self._embeddings.astype("float32"))
                logger.info("[RAGVectorStore] IVF_PQ training complete")

            self._index.add(embeddings.astype("float32"))
            return

        if isinstance(self._index, (faiss.IndexFlatL2, faiss.IndexFlatIP)):
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

    def get_statistics(self) -> Dict[str, Any]:
        """
        FIX (added -- was missing entirely, referenced by
        RAGSchemaMatcher.get_statistics() but never implemented here, so it
        would have raised AttributeError the moment anyone called it).
        Gives exactly the breakdown needed to diagnose "0 results" cases at a
        glance: is the index empty, or are documents present but not tagged
        the way filters expect?
        """
        kind_counts: Dict[str, int] = {}
        for doc in self._documents:
            kind = doc.metadata.get("kind", "unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

        return {
            "dimension": self._dimension,
            "metric": self._metric,
            "requested_index_type": self._index_type_param,
            "actual_index_type": self._actual_index_type,
            "total_documents": len(self._documents),
            "ntotal": int(self._index.ntotal) if self._index is not None else 0,
            "documents_by_kind": kind_counts,
        }

    def clear(self) -> None:
        self._index = None
        self._documents = []
        self._embeddings = None
        self._actual_index_type = None
        logger.info("[RAGVectorStore] Store cleared")