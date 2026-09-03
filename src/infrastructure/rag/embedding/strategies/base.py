from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingStrategy(ABC):
    """
    Strategy interface for text embedding methods.

    Every embedding method must expose the same API to the
    rest of the RAG pipeline.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """
        Encode texts into vectors.
        """
        raise NotImplementedError

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]

    @property
    def name(self) -> str:
        return self.__class__.__name__