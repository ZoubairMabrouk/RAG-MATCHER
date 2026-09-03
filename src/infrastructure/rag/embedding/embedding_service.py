from typing import Sequence

from .strategies.base import EmbeddingStrategy


class EmbeddingService:

    def __init__(
        self,
        strategy: EmbeddingStrategy,
    ):
        self.strategy = strategy

    @property
    def dimension(self) -> int:
        return self.strategy.dimension

    @property
    def method(self) -> str:
        return self.strategy.name

    def encode(
        self,
        texts: Sequence[str],
    ):
        return self.strategy.encode(texts)

    def encode_one(
        self,
        text: str,
    ):
        return self.strategy.encode_one(text)