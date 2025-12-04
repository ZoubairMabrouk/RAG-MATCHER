"""LLM client implementations."""

import json
from typing import Dict, Any, Optional
from openai import OpenAI, api_key, base_url
import anthropic
from src.domain.repositeries.interfaces import ILLMClient
from src.domain.entities.evolution import EvolutionPlan,SchemaChange
from src.domain.entities.schema import ChangeType

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
        
        
class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama", model: str = "phi3:mini"):
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
        text = self.ask(prompt, temperature=temperature, max_tokens=max_tokens)
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
     