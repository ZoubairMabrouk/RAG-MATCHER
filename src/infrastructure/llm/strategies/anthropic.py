from typing import Optional

import anthropic

from .base import LLMStrategy


class AnthropicStrategy(LLMStrategy):

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        temperature: float = 0.1,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

        self.client = anthropic.Anthropic(
            api_key=api_key
        )

    def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or 4096,
            temperature=(
                self.temperature
                if temperature is None
                else temperature
            ),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        if not response.content:
            raise RuntimeError(
                "Anthropic returned an empty response."
            )

        return response.content[0].text.strip()