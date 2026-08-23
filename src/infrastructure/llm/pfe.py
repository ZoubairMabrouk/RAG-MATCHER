from typing import Dict, Any
from abc import ABC

from openai import OpenAI


class BaseLLMClient(ABC):
    """Base LLM client with common functionality."""
    
    def __init__(self, model: str, temperature: float = 0.1):
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API (to be implemented by subclasses)."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature
            )
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return ""




def main():

    client = BaseLLMClient(model="phi3:mini", temperature=0.2)


    prompt = "Explain in one sentence what a database schema is."

    response = client._call_llm(prompt)

    print("\n=== LLM RESPONSE ===")
    print(response)


if __name__ == "__main__":
    main()