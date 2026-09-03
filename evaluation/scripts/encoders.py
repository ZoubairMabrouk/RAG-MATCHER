"""
Encoder interface + adapters — Étape 6.

TFIDFEncoder is fully functional offline (pure scikit-learn, already a
project dependency). SentenceBERTEncoder / BioBERTEncoder / ClinicalBERTEncoder
require downloading pretrained weights from huggingface.co, which is outside
this sandbox's network allowlist (see docs/experimental_architecture.md and
the PHASE reports). Their `.encode()` raises EncoderUnavailableError with a
clear reason rather than being silently skipped or faked.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EncoderUnavailableError(RuntimeError):
    def __init__(self, model_name: str, reason: str, required_dependency: str):
        self.model_name = model_name
        self.reason = reason
        self.required_dependency = required_dependency
        super().__init__(f"{model_name} unavailable: {reason}")


@dataclass
class EncodedResult:
    vectors: np.ndarray
    encoding_time_ms: float


class Encoder(ABC):
    model_name: str

    @abstractmethod
    def encode(self, texts: List[str]) -> EncodedResult:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class TFIDFEncoder(Encoder):
    """Fits a fresh TF-IDF vectorizer on the corpus passed to fit(); this is
    a corpus-dependent encoder (unlike the pretrained transformer encoders),
    so it must be fit once on the combined source+target vocabulary before
    encode() is called for retrieval."""

    model_name = "tfidf"

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._fitted = False
        self._dim = 0

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True
        self._dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: List[str]) -> EncodedResult:
        if not self._fitted:
            raise RuntimeError("TFIDFEncoder.fit(corpus) must be called before encode().")
        start = time.time()
        vectors = self._vectorizer.transform(texts).toarray()
        elapsed_ms = (time.time() - start) * 1000
        return EncodedResult(vectors=vectors, encoding_time_ms=elapsed_ms)

    @property
    def dimension(self) -> int:
        return self._dim


class _UnavailableTransformerEncoder(Encoder):
    """Shared behavior for SBERT/BioBERT/ClinicalBERT: attempts a real
    sentence-transformers load; if the model weights aren't reachable
    (no network to huggingface.co in this sandbox), raises
    EncoderUnavailableError with the exact reason instead of fabricating
    output."""

    hf_model_id: str = ""

    def __init__(self):
        self._model = None

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer  # noqa
        except ImportError as e:
            raise EncoderUnavailableError(
                self.model_name,
                reason=f"sentence-transformers package not installed ({e}).",
                required_dependency="sentence-transformers",
            )
        try:
            self._model = __import__("sentence_transformers").SentenceTransformer(self.hf_model_id)
        except Exception as e:
            raise EncoderUnavailableError(
                self.model_name,
                reason=f"Could not load pretrained weights for '{self.hf_model_id}': {e}. "
                       "This sandbox has no network access to huggingface.co.",
                required_dependency=f"network access to download {self.hf_model_id}",
            )

    def encode(self, texts: List[str]) -> EncodedResult:
        if self._model is None:
            self._try_load()
        start = time.time()
        vectors = self._model.encode(texts)
        elapsed_ms = (time.time() - start) * 1000
        return EncodedResult(vectors=np.asarray(vectors), encoding_time_ms=elapsed_ms)

    @property
    def dimension(self) -> int:
        if self._model is None:
            self._try_load()
        return self._model.get_sentence_embedding_dimension()


class SentenceBERTEncoder(_UnavailableTransformerEncoder):
    model_name = "sbert"
    hf_model_id = "sentence-transformers/all-MiniLM-L6-v2"


class BioBERTEncoder(_UnavailableTransformerEncoder):
    model_name = "biobert"
    hf_model_id = "dmis-lab/biobert-base-cased-v1.2"


class ClinicalBERTEncoder(_UnavailableTransformerEncoder):
    model_name = "clinicalbert"
    hf_model_id = "emilyalsentzer/Bio_ClinicalBERT"


ENCODER_REGISTRY = {
    "tfidf": TFIDFEncoder,
    "sbert": SentenceBERTEncoder,
    "biobert": BioBERTEncoder,
    "clinicalbert": ClinicalBERTEncoder,
}


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cosine_similarity(a, b)