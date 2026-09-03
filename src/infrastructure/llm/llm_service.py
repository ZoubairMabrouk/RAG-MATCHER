import json
import re
from typing import Any, Dict, Optional

from src.domain.entities.evolution import (
    EvolutionPlan,
    SchemaChange,
)
from src.domain.entities.schema import ChangeType

from .strategies.base import LLMStrategy


class LLMService:
    """
    Application-level LLM service.

    Uses a Strategy for provider communication.
    """

    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    @property
    def model(self) -> str:
        return self.strategy.model

    @property
    def provider(self) -> str:
        return self.strategy.provider_name()

    # ---------------------------------------------------------
    # Raw LLM
    # ---------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:

        return self.strategy._call_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ---------------------------------------------------------
    # Schema matching
    # ---------------------------------------------------------

    def choose_best_match(
        self,
        source: Dict[str, Any],
        candidates: list[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not candidates:
            return {
                "best_match": None,
                "confidence": "low",
                "reason": "No candidates available.",
            }

        prompt = f"""
You are a schema alignment expert.

Given the following U-Schema element:

{json.dumps(source, indent=2)}

And these top candidate matches from the database:

{json.dumps(candidates, indent=2)}

Contextual information:

{context or "No extra context."}

Task:

1. Select the single best matching candidate.
2. Explain briefly why it is the best match.
3. Return ONLY valid JSON.

Required format:

{{
    "best_match": "candidate_name",
    "confidence": "high|medium|low",
    "reason": "your reasoning"
}}
"""

        response = self._call_llm(prompt)

        result = self._parse_json(response)

        if result is None:
            return {
                "best_match": candidates[0]["name"],
                "confidence": "low",
                "reason": "Fallback to top similarity candidate.",
            }

        return result

    # ---------------------------------------------------------
    # Evolution
    # ---------------------------------------------------------

    def generate_evolution_plan(
        self,
        context: Dict[str, Any],
    ) -> EvolutionPlan:

        prompt = self._build_evolution_prompt(context)

        response = self._call_llm(
            prompt,
            max_tokens=4096,
        )

        data = self._parse_json(response)

        if data is None:
            raise ValueError(
                "Failed to parse LLM evolution plan response."
            )

        try:
            changes = [
                SchemaChange(
                    change_type=ChangeType(c["change_type"]),
                    target_table=c["target_table"],
                    target_column=c.get("target_column"),
                    definition=c.get("definition"),
                    reason=c["reason"],
                    sql=c.get("sql"),
                    safe=c.get("safe", True),
                    requires_data_migration=c.get(
                        "requires_data_migration",
                        False,
                    ),
                    estimated_impact=c.get(
                        "estimated_impact",
                        "low",
                    ),
                )
                for c in data.get("changes", [])
            ]

            return EvolutionPlan(
                changes=changes,
                description=data.get(
                    "description",
                    "",
                ),
                risk_level=data.get(
                    "risk_level",
                    "low",
                ),
                backward_compatible=data.get(
                    "backward_compatible",
                    True,
                ),
                rollback_plan=data.get(
                    "rollback_plan"
                ),
            )

        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid evolution plan returned by LLM: {exc}"
            ) from exc

    # ---------------------------------------------------------
    # SQL
    # ---------------------------------------------------------

    def generate_sql(
        self,
        change: SchemaChange,
    ) -> str:

        prompt = f"""
Generate PostgreSQL DDL for this schema change.

Change Type:
{change.change_type.value}

Table:
{change.target_table}

Column:
{change.target_column or "N/A"}

Definition:
{change.definition or "N/A"}

Return ONLY the SQL statement.
"""

        return self._call_llm(
            prompt,
            temperature=0.0,
        ).strip()

    # ---------------------------------------------------------
    # Prompt
    # ---------------------------------------------------------

    def _build_evolution_prompt(
        self,
        context: Dict[str, Any],
    ) -> str:

        return f"""
You are a database schema evolution expert.

U-Schema:

{json.dumps(context.get("uschema", {}), indent=2)}

Current Database Schema:

{json.dumps(context.get("current_schema", {}), indent=2)}

Relevant RAG Context:

{json.dumps(context.get("rag_context", {}), indent=2)}

Design Rules:

{json.dumps(context.get("rules", {}), indent=2)}

Task:

Generate a detailed evolution plan to align the
database with the U-Schema.

For each change:

1. Explain WHY it is needed.
2. Assess risk.
3. State whether data migration is required.
4. Provide PostgreSQL DDL.

Return ONLY valid JSON:

{{
    "description": "Overall summary",
    "risk_level": "low|medium|high|critical",
    "changes": [
        {{
            "change_type": "create_table|add_column|etc",
            "target_table": "table_name",
            "target_column": "column_name or null",
            "definition": "SQL definition",
            "reason": "Explanation",
            "sql": "Complete SQL statement",
            "safe": true,
            "requires_data_migration": false,
            "estimated_impact": "low|medium|high"
        }}
    ],
    "backward_compatible": true,
    "rollback_plan": "Steps to rollback"
}}
"""

    # ---------------------------------------------------------
    # JSON parsing
    # ---------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:

        if not text:
            return None

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove markdown fences
        cleaned = re.sub(
            r"```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Extract JSON object
        match = re.search(
            r"\{[\s\S]*\}",
            cleaned,
        )

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None