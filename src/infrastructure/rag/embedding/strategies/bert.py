from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .base import EmbeddingStrategy


class BERTStrategy(EmbeddingStrategy):

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str | None = None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        self._dimension = (
            self.model.config.hidden_size
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = self.model(
                **encoded
            )

        token_embeddings = outputs.last_hidden_state

        attention_mask = encoded[
            "attention_mask"
        ].unsqueeze(-1)

        masked_embeddings = (
            token_embeddings
            * attention_mask
        )

        summed = masked_embeddings.sum(
            dim=1
        )

        counts = attention_mask.sum(
            dim=1
        ).clamp(min=1)

        embeddings = summed / counts

        return (
            embeddings
            .cpu()
            .numpy()
            .astype(np.float32)
            .tolist()
        )