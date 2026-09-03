from typing import Any

from .strategies.tfidf import TFIDFStrategy
from .strategies.word2vec import Word2VecStrategy
from .strategies.bert import BERTStrategy
from .strategies.biobert import BioBERTStrategy
from .strategies.clinicalbert import ClinicalBERTStrategy


class EmbeddingFactory:

    @staticmethod
    def create(
        method: str,
        **kwargs: Any,
    ):

        method = method.lower().strip()

        if method == "tfidf":
            return TFIDFStrategy(
                max_features=kwargs.get(
                    "max_features",
                    5000,
                ),
                ngram_range=kwargs.get(
                    "ngram_range",
                    (1, 2),
                ),
            )

        if method == "word2vec":
            model_path = kwargs.get(
                "model_path"
            )

            if not model_path:
                raise ValueError(
                    "model_path is required for Word2Vec."
                )

            return Word2VecStrategy(
                model_path=model_path
            )

        if method == "bert":
            return BERTStrategy(
                model_name=kwargs.get(
                    "model_name",
                    "bert-base-uncased",
                ),
                device=kwargs.get("device"),
            )

        if method == "biobert":
            return BioBERTStrategy(
                model_name=kwargs.get(
                    "model_name",
                    BioBERTStrategy.DEFAULT_MODEL,
                ),
                device=kwargs.get("device"),
            )

        if method == "clinicalbert":
            return ClinicalBERTStrategy(
                model_name=kwargs.get(
                    "model_name",
                    ClinicalBERTStrategy.DEFAULT_MODEL,
                ),
                device=kwargs.get("device"),
            )

        raise ValueError(
            f"Unsupported embedding method: {method}. "
            "Supported methods: "
            "tfidf, word2vec, bert, biobert, clinicalbert."
        )