from typing import Optional

from openai import OpenAI

from .base import LLMStrategy


class OllamaStrategy(LLMStrategy):
    """
    LLM strategy for local Ollama models.

    Ollama exposes an OpenAI-compatible API.
    """

    DEFAULT_BASE_URL = "http://localhost:11435/v1"

    def __init__(
        self,
        model: str = "phi3:mini",
        temperature: float = 0.1,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "ollama",
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:

        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": (
                self.temperature
                if temperature is None
                else temperature
            ),
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                f"Ollama returned an empty response for model '{self.model}'."
            )

        return content.strip()