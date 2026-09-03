from typing import List, Sequence

import numpy as np

from .base import EmbeddingStrategy


class Word2VecStrategy(EmbeddingStrategy):

    def __init__(
        self,
        model_path: str,
    ):
        from gensim.models import Word2Vec

        self.model = Word2Vec.load(
            model_path
        )

        self._dimension = self.model.vector_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:

        embeddings = []

        for text in texts:

            tokens = text.lower().split()

            vectors = [
                self.model.wv[token]
                for token in tokens
                if token in self.model.wv
            ]

            if not vectors:
                vector = np.zeros(
                    self.dimension,
                    dtype=np.float32,
                )
            else:
                vector = np.mean(
                    vectors,
                    axis=0,
                ).astype(np.float32)

            embeddings.append(
                vector.tolist()
            )

        return embeddings