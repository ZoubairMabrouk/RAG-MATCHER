from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMStrategy(ABC):
    """
    Strategy interface for LLM providers.

    The strategy is responsible only for communicating with
    a specific LLM provider.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
    ):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Execute a prompt against the provider.
        """
        raise NotImplementedError

    def supports_json_mode(self) -> bool:
        return False

    def provider_name(self) -> str:
        return self.__class__.__name__