"""
Unified Embedding Services for RAG.

This module provides:
- EmbeddingProvider protocol
- LocalEmbeddingProvider (Sentence-Transformers)
- OpenAIEmbeddingProvider (optional; requires openai)
- RAGEmbeddingService (bi-encoder + cross-encoder with reranking, cosine sims)
- Legacy EmbeddingService wrapper for backward compatibility

Keep FAISS/vector store dimension in sync with the provider/service dimension.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Protocol, Optional
import logging
import os
import numpy as np

# Domain types (adjust paths if your project differs)
from src.domain.entities.rag_schema import (
    KnowledgeBaseDocument,
    SourceField,
    RetrievalQuery,
)

logger = logging.getLogger(__name__)

# Import with fallback stubs for testing
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:  # fallback minimal si package non dispo
    class CrossEncoder:  # stub pour tests
        def __init__(self, *_, **__): 
            pass
        def predict(self, pairs): 
            # renvoyer un score constant ou basé sur len(pairs) ; suffit pour tests
            return np.ones(len(pairs), dtype=float)
    
    class SentenceTransformer:
        def __init__(self, *_, **__): 
            pass
        def get_sentence_embedding_dimension(self): 
            return 384
        def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
            # stub: vecteurs unitaires dimension 384
            arr = np.zeros((len(texts), 384), dtype="float32")
            if normalize_embeddings: 
                return arr
            return arr


# ---------------------------------------------------------------------------
# Provider protocol and implementations
# ---------------------------------------------------------------------------

class EmbeddingProvider(Protocol):
    """Protocol for simple embedding providers (bi-encoder style)."""
    model: str
    dimension: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class LocalEmbeddingProvider:
    """
    Offline embeddings via sentence-transformers.
    Default model: all-MiniLM-L6-v2 (384 dims).
    """
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        logger.info(f"[LocalEmbeddingProvider] Loading model: {model}")
        self.model = model
        self._encoder = SentenceTransformer(model)
        self.dimension = int(self._encoder.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Normalize for cosine-friendly behavior
        emb = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()


class OpenAIEmbeddingProvider:
    """
    OpenAI embeddings (online). Default: text-embedding-3-small (1536 dims).

    NOTE: Ensure dimension matches your vector index (e.g., FAISS).
    - text-embedding-3-small  -> 1536
    - text-embedding-3-large  -> 3072
    """
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        api_key="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"
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "openai package is required for OpenAIEmbeddingProvider. "
                "Install with: pip install openai"
            ) from e

        self._client = OpenAI(api_key=api_key)
        self.model = model
        # Choose dimension based on model
        self.dimension = 3072 if "large" in model else 1536

        logger.info(f"[OpenAIEmbeddingProvider] Using model: {model} (dim={self.dimension})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# RAG Embedding Service (bi-encoder + cross-encoder)
# ---------------------------------------------------------------------------

class RAGEmbeddingService:
    """
    Advanced embedding service for Retrieval-Augmented Generation:
    - Bi-encoder (Sentence-Transformers) for fast retrieval vectors.
    - Cross-encoder for accurate reranking of top candidates.

    Single responsibility: embedding generation and similarity scoring.
    """

    def __init__(
        self,
        bi_encoder_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        show_progress: bool = False,
    ):
        """
        Args:
            bi_encoder_model: sentence-transformers model for retrieval
            cross_encoder_model: cross-encoder model for reranking
            show_progress: show progress bars on encode
        """
        logger.info(f"[RAGEmbeddingService] Loading bi-encoder: {bi_encoder_model}")
        self._bi = SentenceTransformer(bi_encoder_model)

        logger.info(f"[RAGEmbeddingService] Loading cross-encoder: {cross_encoder_model}")
        self._cross = CrossEncoder(cross_encoder_model)

        self._show_progress = show_progress
        self.dimension = int(self._bi.get_sentence_embedding_dimension())
        logger.info(f"[RAGEmbeddingService] Embedding dimension: {self.dimension}")

    # ---- Document & query embeddings ------------------------------------------------

    def embed_documents(self, documents: List[KnowledgeBaseDocument]) -> np.ndarray:
        texts = [doc.to_text() for doc in documents]
        logger.info(f"[RAGEmbeddingService] Embedding {len(texts)} documents")
        embeddings = self._bi.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: RetrievalQuery) -> np.ndarray:
        query_text = self._build_query_text(query)
        emb = self._bi.encode([query_text], normalize_embeddings=True)
        return emb[0].astype("float32")

    # ---- Reranking & similarity -----------------------------------------------------

    def rerank_candidates(
        self,
        query: RetrievalQuery,
        candidates: List[KnowledgeBaseDocument],
    ) -> List[Tuple[KnowledgeBaseDocument, float]]:
        if not candidates:
            return []

        query_text = self._build_query_text(query)
        pairs = [(query_text, c.to_text()) for c in candidates]
        scores = self._cross.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Cosine similarity between a single query vector and N candidate vectors.
        All vectors are normalized for numerical stability.
        """
        q = query_embedding
        Q = q / (np.linalg.norm(q) + 1e-12)
        C = candidate_embeddings
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        return np.dot(Cn, Q)

    # ---- Helpers -------------------------------------------------------------------

    def _build_query_text(self, query: RetrievalQuery) -> str:
        """Flatten a RetrievalQuery into a searchable text prompt."""
        sf: SourceField = query.source_field

        parts: List[str] = [
            f"Field: {sf.path}",
            f"Name tokens: {', '.join(sf.name_tokens)}",
            f"Type: {sf.inferred_type}",
        ]

        if getattr(sf, "format_regex", None):
            parts.append(f"Format: {sf.format_regex}")
        if getattr(sf, "units", None):
            parts.append(f"Units: {sf.units}")
        if getattr(sf, "category_values", None):
            vals = sf.category_values[:3] if sf.category_values else []
            if vals:
                parts.append(f"Sample values: {', '.join(vals)}")
        if getattr(sf, "hints", None):
            parts.append(f"Hints: {', '.join(sf.hints)}")
        if getattr(sf, "neighbors", None):
            parts.append(f"Neighbors: {', '.join(sf.neighbors)}")
        if getattr(sf, "coarse_semantics", None):
            parts.append(f"Semantics: {', '.join(sf.coarse_semantics)}")

        return ". ".join(parts)

    def get_embedding_dimension(self) -> int:
        return self.dimension


# ---------------------------------------------------------------------------
# Legacy wrapper for backward compatibility
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Legacy embedding service wrapper used by older code paths.
    Delegates to a simple EmbeddingProvider (Local or OpenAI).
    """

    def __init__(self, provider: EmbeddingProvider):
        self._provider = provider

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Simple passthrough to the provider."""
        return self._provider.embed(texts)

    # ---- Legacy helpers for schema objects -----------------------------------------

    def embed_schema_objects(self, schema) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        Generate embeddings for all schema objects.
        Returns tuples of (text, embedding_vector, metadata).
        """
        results: List[Tuple[str, np.ndarray, Dict]] = []

        for table in getattr(schema, "tables", []):
            # Table-level
            t_text = self._table_to_text(table)
            t_emb = np.array(self._provider.embed([t_text])[0], dtype="float32")
            t_meta: Dict = {
                "type": "table",
                "name": getattr(table, "name", None),
                "schema": getattr(table, "schema", None),
                "row_count": getattr(table, "row_count", None),
            }
            results.append((t_text, t_emb, t_meta))

            # Column-level
            for col in getattr(table, "columns", []):
                c_text = self._column_to_text(getattr(table, "name", "?"), col)
                c_emb = np.array(self._provider.embed([c_text])[0], dtype="float32")
                c_meta: Dict = {
                    "type": "column",
                    "table": getattr(table, "name", None),
                    "name": getattr(col, "name", None),
                    "data_type": getattr(col, "data_type", None),
                }
                results.append((c_text, c_emb, c_meta))

        return results

    def _table_to_text(self, table) -> str:
        cols = ", ".join([getattr(c, "name", "?") for c in getattr(table, "columns", [])])
        comment = getattr(table, "comment", None) or "None"
        return f"Table: {getattr(table, 'name', '?')}. Columns: {cols}. Comment: {comment}"

    def _column_to_text(self, table_name: str, col) -> str:
        comment = getattr(col, "comment", None) or "None"
        return (
            f"Column: {table_name}.{getattr(col, 'name', '?')}. "
            f"Type: {getattr(col, 'data_type', '?')}. Comment: {comment}"
        )


# ---------------------------------------------------------------------------
# Small factory helpers (optional)
# ---------------------------------------------------------------------------

def build_default_provider() -> EmbeddingProvider:
    """
    Create a default provider from environment:
      - RAG_EMBED_PROVIDER=local|openai (default: local)
      - LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2 (for local)
      - OPENAI_API_KEY + EMBEDDING_MODEL=text-embedding-3-small (for openai)
    """
    provider = os.getenv("RAG_EMBED_PROVIDER", "local").lower()
    if provider == "openai":
        api_key ="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" #os.getenv("OPENAI_API_KEY", "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")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)
    else:
        model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return LocalEmbeddingProvider(model=model)


__all__ = [
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "RAGEmbeddingService",
    "EmbeddingService",
    "build_default_provider",
]
