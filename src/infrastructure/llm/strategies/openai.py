from typing import Optional

from openai import OpenAI

from .base import LLMStrategy


class OpenAIStrategy(LLMStrategy):

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.DEFAULT_BASE_URL,
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
            raise RuntimeError("OpenAI returned an empty response.")

        return content.strip()