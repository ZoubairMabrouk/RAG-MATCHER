import os
from typing import Any

from .llm_service import LLMService
from .strategies.ollama import OllamaStrategy
from .strategies.openai import OpenAIStrategy
from .strategies.anthropic import AnthropicStrategy
from .strategies.gemini import GeminiStrategy


class LLMFactory:

    @staticmethod
    def create(
        provider: str,
        **kwargs: Any,
    ) -> LLMService:

        provider = provider.lower().strip()

        if provider == "ollama":
            strategy = OllamaStrategy(
                model=kwargs.get(
                    "model",
                    "phi3:mini",
                ),
                temperature=kwargs.get(
                    "temperature",
                    0.1,
                ),
                base_url=kwargs.get(
                    "base_url",
                    "http://localhost:11434/v1",
                ),
            )

        elif provider == "openai":
            api_key = kwargs.get(
                "api_key"
            ) or os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required."
                )

            strategy = OpenAIStrategy(
                api_key=api_key,
                model=kwargs.get(
                    "model",
                    "gpt-4o",
                ),
                temperature=kwargs.get(
                    "temperature",
                    0.1,
                ),
            )

        elif provider == "anthropic":
            api_key = kwargs.get(
                "api_key"
            ) or os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required."
                )

            strategy = AnthropicStrategy(
                api_key=api_key,
                model=kwargs.get(
                    "model",
                    "claude-3-5-sonnet-latest",
                ),
                temperature=kwargs.get(
                    "temperature",
                    0.1,
                ),
            )

        elif provider == "gemini":
            # api_key can be a single string OR a list of strings (rotation).
            # Resolution order:
            #   1. explicit kwargs["api_key"] (str or list)
            #   2. GEMINI_API_KEYS env var, comma-separated -> list (rotation)
            #   3. GEMINI_API_KEY env var -> single key
            api_key = [
            "AQ.Ab8RN6Kt6_b5HLI687ijb9MLOD4o2Jcay1JKe556O-OpfWFArw",
            "AQ.Ab8RN6IHXb6IXnpEakhMctmU1mt3yDFLjvjEqRMwez-FAjM7Uw",
            "AQ.Ab8RN6JSWB8YmdaKJCqWR_Ppf8X_i8qVCBL3aUWxdJgIr4iOVw",
            "AQ.Ab8RN6J9SO6bCkix7D754F7WYRPq5RDs3e0N9h9RRMX2Zo3E-A"
        ]
 
            strategy = GeminiStrategy(
                api_key=api_key,
                model="gemini-3.7-flash",
                temperature=kwargs.get("temperature", 0.1),
                max_retries_per_key=kwargs.get("max_retries_per_key", 1),
            )
 
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: ollama, openai, anthropic, gemini."
            )
 
        return LLMService(strategy)