"""LLM client implementations."""

import json
from typing import Dict, Any, List, Optional
from openai import OpenAI, api_key, base_url
import anthropic
from src.domain.repositeries.interfaces import ILLMClient
from src.domain.entities.evolution import EvolutionPlan,SchemaChange
from src.domain.entities.schema import ChangeType

import logging
import os
import time

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


log = logging.getLogger(__name__)
class BaseLLMClient(ILLMClient):
    """Base LLM client with common functionality."""
    
    def __init__(self, model: str, temperature: float = 0.1):
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    def _build_evolution_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for evolution plan generation."""
        print("prompt building...")
        prompt = """You are a database schema evolution expert. Analyze the following:

U-Schema (Target):
{uschema}

Current Database Schema:
{current_schema}

Relevant Context (from RAG):
{rag_context}

Design Rules:
{rules}

Task: Generate a detailed evolution plan to align the database with the U-Schema.
For each change:
1. Explain WHY it's needed
2. Assess the risk (low/medium/high)
3. Note if data migration is required
4. Provide the SQL DDL

Return your response as valid JSON with this structure:
{{
  "description": "Overall summary of changes",
  "risk_level": "low|medium|high|critical",
  "changes": [
    {{
      "change_type": "create_table|add_column|etc",
      "target_table": "table_name",
      "target_column": "column_name or null",
      "definition": "SQL definition",
      "reason": "Explanation",
      "sql": "Complete SQL statement",
      "safe": true/false,
      "requires_data_migration": true/false,
      "estimated_impact": "low|medium|high"
    }}
  ],
  "backward_compatible": true/false,
  "rollback_plan": "Steps to rollback if needed"
}}"""
        
        return prompt.format(
            uschema=json.dumps(context.get("uschema", {}), indent=2),
            current_schema=json.dumps(context.get("current_schema", {}), indent=2),
            rag_context=json.dumps(context.get("rag_context", {}), indent=2),
            rules=json.dumps(context.get("rules", {}), indent=2)
        )
    
    def generate_evolution_plan(self, context: Dict[str, Any]) -> EvolutionPlan:
        """Generate evolution plan using LLM."""
        prompt = self._build_evolution_prompt(context)
        response = self._call_llm(prompt)
        
        # Parse response
        try:
            data = json.loads(response)
            changes = [
                SchemaChange(
                    change_type=ChangeType(c["change_type"]),
                    target_table=c["target_table"],
                    target_column=c.get("target_column"),
                    definition=c.get("definition"),
                    reason=c["reason"],
                    sql=c.get("sql"),
                    safe=c.get("safe", True),
                    requires_data_migration=c.get("requires_data_migration", False),
                    estimated_impact=c.get("estimated_impact", "low")
                )
                for c in data["changes"]
            ]
            
            return EvolutionPlan(
                changes=changes,
                description=data["description"],
                risk_level=data.get("risk_level", "low"),
                backward_compatible=data.get("backward_compatible", True),
                rollback_plan=data.get("rollback_plan")
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}")
    
    def generate_sql(self, change: SchemaChange) -> str:
        """Generate SQL for a schema change."""
        prompt = f"""Generate PostgreSQL DDL for this schema change:

Change Type: {change.change_type.value}
Table: {change.target_table}
Column: {change.target_column or 'N/A'}
Definition: {change.definition or 'N/A'}

Return ONLY the SQL statement, nothing else."""
        
        return self._call_llm(prompt).strip()
    
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

    def choose_best_match(self, source: Dict[str, Any], candidates: list[Dict[str, Any]], context: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask LLM to pick the best matching candidate from top-K retrieved.
        """
        prompt = f"""
    You are a schema alignment expert.

    Given the following U-Schema element:
    {json.dumps(source, indent=2)}

    And these top candidate matches from the database:
    {json.dumps(candidates, indent=2)}

    Contextual info:
    {context or "No extra context."}

    Task:
    - Select the *single best matching candidate*.
    - Explain briefly why you chose it.
    - Return valid JSON in the format:
    {{
    "best_match": "candidate_name",
    "confidence": "high|medium|low",
    "reason": "your reasoning"
    }}
    """
        response = self._call_llm(prompt)

        try:
            result = json.loads(response)
            return result
        except Exception as e:
            print(f"⚠️ Failed to parse LLM response: {e}")
            # fallback: pick top candidate
            return {
                "best_match": candidates[0]["name"],
                "confidence": "low",
                "reason": "Fallback to top similarity score."
            }


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM client implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        super().__init__(model)
        self._api_key = api_key
        self._client = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1")
    
    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API."""
        # Implementation would use openai library
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return ""    
        # Placeholder
        #return '{"description": "Sample", "changes": [], "risk_level": "low"}'

class AnthropicLLMClient(BaseLLMClient):
    """Anthropic LLM client implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(model)
        self._api_key = api_key
    
    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic API."""
        # Implementation would use anthropic library
        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        
class GeminiLLMClient(BaseLLMClient):
    """Google Gemini LLM client implementation."""

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.7-flash",
        temperature: float = 0.1,
    ):
        super().__init__(model=model, temperature=temperature)

        self._api_key = api_key

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self.GEMINI_BASE_URL,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call Google Gemini API."""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=self._temperature,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Gemini returned an empty response.")

            return content

        except Exception as e:
            print(f"Gemini API call failed: {e}")
            raise
class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11435/v1", api_key: str = "ollama", model: str = "phi3:mini"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in database schema matching and mapping."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def ask_json(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> Dict[str, Any]:
        """Ask LLM and parse JSON. Caller should handle exceptions/fallbacks."""
        text = self._call_llm(prompt, temperature=temperature, max_tokens=max_tokens)
        # Robust parse: try json.loads, else try to extract JSON substring
        try:
            return json.loads(text)
        except Exception:
            # attempt to salvage JSON-like section
            import re
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            # If still failing, return a fallback mapping
            return {"selected": None, "confidence": 0.0, "rationale": text}



class GeminiKeyPool:
    """
    Manages a list of Gemini API keys.

    A key is marked unavailable after a quota/authentication error.
    The pool then automatically selects the next available key.
    """

    def __init__(
        self,
    ):
        self.api_keys = [
            "AQ.Ab8RN6Kt6_b5HLI687ijb9MLOD4o2Jcay1JKe556O-OpfWFArw",
            "AQ.Ab8RN6IHXb6IXnpEakhMctmU1mt3yDFLjvjEqRMwez-FAjM7Uw",
            "AQ.Ab8RN6JSWB8YmdaKJCqWR_Ppf8X_i8qVCBL3aUWxdJgIr4iOVw",
            "AQ.Ab8RN6J9SO6bCkix7D754F7WYRPq5RDs3e0N9h9RRMX2Zo3E-A"
        ]

        if not self.api_keys:
            raise ValueError(
                "No Gemini API keys were provided."
            )

        self._current_index = 0
        self._exhausted_indices = set()

    # --------------------------------------------------------
    # Current key
    # --------------------------------------------------------

    @property
    def current_key(self) -> Optional[str]:

        if self.exhausted:
            return None

        return self.api_keys[self._current_index]

    @property
    def current_key_number(self) -> int:
        return self._current_index + 1

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        return len(self._exhausted_indices) >= len(self.api_keys)

    @property
    def remaining(self) -> int:
        return len(
            self.api_keys
        ) - len(
            self._exhausted_indices
        )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    def mark_current_exhausted(self) -> Optional[str]:

        current = self._current_index

        self._exhausted_indices.add(current)

        if self.exhausted:
            return None

        total = len(self.api_keys)

        for offset in range(1, total + 1):

            candidate = (
                current + offset
            ) % total

            if candidate not in self._exhausted_indices:

                self._current_index = candidate

                return self.api_keys[candidate]

        return None


# ============================================================
# Gemini
# ============================================================

class GeminiLLMClient(BaseLLMClient):
    """
    Gemini client with automatic API-key rotation.

    Environment:

        GEMINI_API_KEYS=key1,key2,key3,key4

    Behavior:

        key1 -> OK
        key1 -> 429
        key2 -> OK
        key2 -> 429
        key3 -> OK

    The key is NOT rotated after every request.
    It is rotated only when the current key becomes unusable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model: str = "gemini-3.6-flash",
        temperature: float = 0.1,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            **kwargs,
        )

        if genai is None:
            raise ImportError(
                "google-genai is not installed. "
                "Install it with: pip install google-genai"
            )

        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # ----------------------------------------------------
        # Load API keys
        # ----------------------------------------------------

        keys = []

        if api_keys:
            keys.extend(api_keys)

        if api_key:
            keys.append(api_key)

        env_keys = os.getenv(
            "GEMINI_API_KEYS",
            "",
        )

        if env_keys:
            keys.extend(
                env_keys.split(",")
            )

        # Remove duplicates while preserving order
        unique_keys = []

        seen = set()

        for key in keys:

            key = key.strip()

            if key and key not in seen:

                unique_keys.append(key)
                seen.add(key)

        self.key_pool = GeminiKeyPool(
            unique_keys
        )

        self._client = None

        self._create_client()

        log.info(
            "Gemini key pool initialized with %d key(s)",
            len(self.key_pool.api_keys),
        )

    # --------------------------------------------------------
    # Client creation
    # --------------------------------------------------------

    def _create_client(self):

        key = self.key_pool.current_key

        if key is None:

            raise LLMQuotaExhaustedError(
                "All Gemini API keys are exhausted."
            )

        self._client = genai.Client(
            api_key=key
        )

        log.info(
            "Gemini using API key #%d",
            self.key_pool.current_key_number,
        )

    # --------------------------------------------------------
    # Error classification
    # --------------------------------------------------------

    @staticmethod
    def _error_text(exc: Exception) -> str:

        return (
            f"{type(exc).__name__}: {exc}"
        ).lower()

    def _is_quota_error(
        self,
        exc: Exception,
    ) -> bool:

        text = self._error_text(exc)

        indicators = [
            "429",
            "too many requests",
            "quota",
            "resource exhausted",
            "rate limit",
            "rate_limit",
            "generate_content_free_tier_requests",
        ]

        return any(
            indicator in text
            for indicator in indicators
        )

    def _is_auth_error(
        self,
        exc: Exception,
    ) -> bool:

        text = self._error_text(exc)

        indicators = [
            "401",
            "403",
            "unauthorized",
            "permission denied",
            "invalid api key",
            "api key not valid",
        ]

        return any(
            indicator in text
            for indicator in indicators
        )

    def _is_retryable_server_error(
        self,
        exc: Exception,
    ) -> bool:

        text = self._error_text(exc)

        indicators = [
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "service unavailable",
        ]

        return any(
            indicator in text
            for indicator in indicators
        )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    def _rotate_key(self) -> bool:

        old_number = (
            self.key_pool.current_key_number
        )

        next_key = (
            self.key_pool.mark_current_exhausted()
        )

        if next_key is None:

            log.error(
                "All %d Gemini API keys are exhausted.",
                len(self.key_pool.api_keys),
            )

            return False

        log.warning(
            "Gemini API key #%d exhausted. "
            "Switching to key #%d.",
            old_number,
            self.key_pool.current_key_number,
        )

        self._create_client()

        return True

    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:

        while not self.key_pool.exhausted:

            try:

                response = (
                    self._client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        **self._build_generation_config(
                            kwargs
                        ),
                    )
                )

                return self._extract_text(
                    response
                )

            except Exception as exc:

                # --------------------------------------------
                # Quota -> rotate key
                # --------------------------------------------

                if self._is_quota_error(exc):

                    log.warning(
                        "Gemini quota exceeded "
                        "for key #%d.",
                        self.key_pool.current_key_number,
                    )

                    if self._rotate_key():

                        continue

                    raise LLMQuotaExhaustedError(
                        "All Gemini API keys have "
                        "exceeded their quota."
                    ) from exc

                # --------------------------------------------
                # Authentication -> rotate key
                # --------------------------------------------

                if self._is_auth_error(exc):

                    log.warning(
                        "Gemini authentication error "
                        "for key #%d.",
                        self.key_pool.current_key_number,
                    )

                    if self._rotate_key():

                        continue

                    raise LLMQuotaExhaustedError(
                        "All Gemini API keys are invalid "
                        "or unavailable."
                    ) from exc

                # --------------------------------------------
                # Temporary server error
                # --------------------------------------------

                if self._is_retryable_server_error(
                    exc
                ):

                    for retry in range(
                        self.max_retries
                    ):

                        delay = (
                            self.retry_delay
                            * (retry + 1)
                        )

                        log.warning(
                            "Gemini temporary error. "
                            "Retry %d/%d in %.1fs.",
                            retry + 1,
                            self.max_retries,
                            delay,
                        )

                        time.sleep(delay)

                        try:

                            response = (
                                self._client.models.generate_content(
                                    model=self.model,
                                    contents=prompt,
                                    **self._build_generation_config(
                                        kwargs
                                    ),
                                )
                            )

                            return self._extract_text(
                                response
                            )

                        except Exception as retry_exc:

                            if self._is_quota_error(
                                retry_exc
                            ):

                                break

                            if not self._is_retryable_server_error(
                                retry_exc
                            ):

                                raise

                    # after retries, rotate only if
                    # quota was reached
                    if self.key_pool.exhausted:
                        break

                    raise

                # --------------------------------------------
                # Other error
                # --------------------------------------------

                log.error(
                    "Gemini request failed: %s",
                    exc,
                )

                raise

        raise LLMQuotaExhaustedError(
            "All Gemini API keys are exhausted."
        )

    # --------------------------------------------------------
    # Chat alias
    # --------------------------------------------------------

    def chat(
        self,
        prompt: str,
        **kwargs,
    ) -> str:

        return self.generate(
            prompt,
            **kwargs,
        )

    # --------------------------------------------------------
    # Generation configuration
    # --------------------------------------------------------

    def _build_generation_config(
        self,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:

        config = {}

        temperature = kwargs.pop(
            "temperature",
            self.temperature,
        )

        config["config"] = {
            "temperature": temperature,
        }

        # Optional max output tokens
        max_output_tokens = kwargs.pop(
            "max_output_tokens",
            None,
        )

        if max_output_tokens is not None:

            config["config"][
                "max_output_tokens"
            ] = max_output_tokens

        return config

    # --------------------------------------------------------
    # Response extraction
    # --------------------------------------------------------

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:

        if response is None:
            return ""

        # google-genai response
        text = getattr(
            response,
            "text",
            None,
        )

        if text:
            return text

        # fallback
        candidates = getattr(
            response,
            "candidates",
            None,
        )

        if candidates:

            parts = getattr(
                candidates[0],
                "content",
                None,
            )

            if parts:

                parts = getattr(
                    parts,
                    "parts",
                    [],
                )

                texts = []

                for part in parts:

                    part_text = getattr(
                        part,
                        "text",
                        None,
                    )

                    if part_text:
                        texts.append(
                            part_text
                        )

                return "".join(texts)

        return str(response)