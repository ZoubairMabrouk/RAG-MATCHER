import logging
import time
from typing import List, Optional, Union

from google import genai
from google.genai import types

from .base import LLMStrategy

logger = logging.getLogger(__name__)


# Substrings that reliably indicate "this specific key is exhausted/invalid",
# as opposed to a transient network error or a real bug elsewhere. Matched
# case-insensitively against str(exception) since the exact exception class
# raised by google-genai varies across SDK versions (ClientError, APIError,
# ResourceExhausted, etc.) -- string matching is more version-robust than
# importing and catching a specific class that might not exist in every
# installed version.
_KEY_EXHAUSTED_MARKERS = (
    "resource_exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "429",
    "permission_denied",
    "api_key_invalid",
    "invalid api key",
    "unauthenticated",
    "401",
    "403",
)


class AllKeysExhaustedError(RuntimeError):
    """Raised when every configured API key has failed with a
    quota/auth-type error -- i.e. rotation ran out of options."""

    def __init__(self, attempts: List[str]):
        self.attempts = attempts
        super().__init__(
            f"All {len(attempts)} Gemini API key(s) failed with quota/auth errors: "
            + "; ".join(attempts)
        )


class GeminiStrategy(LLMStrategy):
    """
    Gemini implementation using the official Google GenAI SDK.

    Supports one or several API keys. When a call fails with a quota/auth
    error (429, RESOURCE_EXHAUSTED, invalid/expired key, ...), the strategy
    automatically rotates to the next key in the list and retries the SAME
    request -- transparent to the caller. Non-quota errors (network blips,
    malformed prompt, etc.) are retried on the CURRENT key with backoff
    instead of burning through the key list, since rotating wouldn't fix them.
    """

    def __init__(
        self,
        api_key: Optional[Union[str, List[str]]] = None,
        model: str = "gemini-3.7-flash",
        temperature: float = 0.1,
        max_retries_per_key: int = 1,
    ):
        super().__init__(model=model, temperature=temperature)

        keys = self._normalize_keys(api_key)
        if not keys:
            raise ValueError(
                "At least one Gemini API key is required. Pass api_key= as a "
                "single string or a list of strings, or set GEMINI_API_KEY "
                "(and optionally GEMINI_API_KEYS, comma-separated, for rotation)."
            )

        self._api_keys: List[str] = keys
        self._key_index: int = 0
        self.max_retries_per_key = max_retries_per_key

        self.api_key = self._api_keys[self._key_index]  # kept for backward compat / introspection
        self.gemini_client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _normalize_keys(api_key: Optional[Union[str, List[str]]]) -> List[str]:
        if api_key is None:
            return []
        if isinstance(api_key, str):
            keys = [api_key]
        else:
            keys = list(api_key)
        seen = set()
        result = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                result.append(k)
        return result

    def supports_json_mode(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def _current_key_label(self) -> str:
        """Never log/expose the full key -- only a short, non-reversible hint."""
        key = self._api_keys[self._key_index]
        return f"key#{self._key_index + 1}/{len(self._api_keys)} (...{key[-4:]})"

    def _rotate_to_next_key(self) -> bool:
        """Advances to the next key. Returns False if we've already tried
        every key (caller should give up)."""
        if self._key_index + 1 >= len(self._api_keys):
            return False
        self._key_index += 1
        self.api_key = self._api_keys[self._key_index]
        self.gemini_client = genai.Client(api_key=self.api_key)
        logger.warning("[GeminiStrategy] Rotated to %s", self._current_key_label())
        return True

    @staticmethod
    def _is_key_exhausted_error(e: Exception) -> bool:
        text = str(e).lower()
        return any(marker in text for marker in _KEY_EXHAUSTED_MARKERS)

    # ------------------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        config = types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=max_tokens,
        )

        key_failure_log: List[str] = []

        while True:
            # Retry the CURRENT key a bounded number of times for transient
            # (non-quota) errors before deciding whether to rotate.
            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries_per_key + 1):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                    content = (response.text or "").strip()
                    if content:
                        return content
                    last_error = RuntimeError("Gemini returned an empty response.")
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    if self._is_key_exhausted_error(e):
                        break  # don't waste retries on a dead key -- rotate immediately
                    logger.warning(
                        "[GeminiStrategy] transient error on %s (attempt %d/%d): %s",
                        self._current_key_label(), attempt + 1, self.max_retries_per_key + 1, e,
                    )
                    if attempt < self.max_retries_per_key:
                        time.sleep(2 ** attempt)

            key_failure_log.append(f"{self._current_key_label()}: {last_error}")

            if last_error is not None and self._is_key_exhausted_error(last_error):
                if self._rotate_to_next_key():
                    continue  # try the same request again on the new key

            if len(key_failure_log) >= len(self._api_keys):
                raise AllKeysExhaustedError(key_failure_log)
            raise RuntimeError(f"Gemini call failed: {last_error}") from last_error