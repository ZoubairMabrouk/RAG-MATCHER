from typing import List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import EmbeddingStrategy


class TFIDFStrategy(EmbeddingStrategy):

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range=(1, 2),
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

        self._dimension = 0
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        self.vectorizer.fit(texts)

        self._dimension = len(
            self.vectorizer.vocabulary_
        )

        self._fitted = True

    @property
    def dimension(self) -> int:
        if not self._fitted:
            raise RuntimeError(
                "TF-IDF strategy must be fitted before "
                "accessing its dimension."
            )

        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:

        if not self._fitted:
            raise RuntimeError(
                "TF-IDF strategy must be fitted before encoding."
            )

        matrix = self.vectorizer.transform(texts)

        return matrix.toarray().astype(
            np.float32
        ).tolist()