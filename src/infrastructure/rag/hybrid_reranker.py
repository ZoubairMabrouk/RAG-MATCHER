"""Schema-aware hybrid scoring for table and column matching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AttributeSpec:
    name: str
    data_type: str = "string"
    is_key: bool = False
    embedding: Optional[np.ndarray] = None


@dataclass
class EntitySpec:
    name: str
    attributes: List[AttributeSpec] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class Decision(str, Enum):
    MATCH = "MATCH"
    REVIEW = "REVIEW"
    NO_MATCH = "NO_MATCH"


@dataclass
class ScoreBreakdown:
    semantic: float
    lexical: float
    columns: float
    types: float
    structure: float
    final: float


@dataclass
class MatchResult:
    source_name: str
    target_name: Optional[str]
    confidence: float
    decision: Decision
    rationale: str
    breakdown: Optional[ScoreBreakdown] = None
    margin: float = 0.0
    top_candidates: List[Tuple[str, float]] = field(default_factory=list)
    column_mapping: Dict[str, Tuple[Optional[str], float]] = field(default_factory=dict)


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")
_TYPE_FAMILIES = {
    "string": {"string", "varchar", "text", "char", "nvarchar", "character varying"},
    "integer": {"integer", "int", "bigint", "smallint", "serial", "bigserial"},
    "float": {"float", "decimal", "double", "real", "numeric"},
    "boolean": {"boolean", "bool"},
    "date": {"date"},
    "timestamp": {"timestamp", "datetime", "timestamptz"},
    "json": {"json", "jsonb"},
    "uuid": {"uuid"},
}
_SOFT_COMPAT = {
    ("date", "timestamp"): 0.85,
    ("timestamp", "date"): 0.85,
    ("integer", "float"): 0.6,
    ("float", "integer"): 0.6,
    ("string", "json"): 0.3,
    ("json", "string"): 0.3,
    ("string", "uuid"): 0.4,
    ("uuid", "string"): 0.4,
}


def _tokens(value: str) -> set[str]:
    value = (value or "").replace("->", " ").replace("_", " ")
    value = _CAMEL_RE.sub(" ", value).lower()
    result = []
    for token in re.findall(r"[a-z0-9]+", value):
        if token.endswith("ies") and len(token) > 3:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        result.append(token)
    return set(result)


def lexical_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_norm = " ".join(sorted(_tokens(left)))
    right_norm = " ".join(sorted(_tokens(right)))
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0
    return 0.6 * overlap + 0.4 * SequenceMatcher(None, left_norm, right_norm).ratio()


def _family(data_type: str) -> Optional[str]:
    value = (data_type or "").strip().lower()
    for family, members in _TYPE_FAMILIES.items():
        if value == family or value in members:
            return family
    return None


def type_similarity(source_type: str, target_type: str) -> float:
    if not source_type or not target_type:
        return 0.5
    if source_type.strip().lower() == target_type.strip().lower():
        return 1.0
    source_family, target_family = _family(source_type), _family(target_type)
    if source_family is None or target_family is None:
        return 0.5
    if source_family == target_family:
        return 1.0
    return _SOFT_COMPAT.get((source_family, target_family), 0.1)


def cosine_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    denominator = np.linalg.norm(query) * np.linalg.norm(candidate)
    return float(np.dot(query, candidate) / denominator) if denominator else 0.0


def column_coverage(source: Sequence[AttributeSpec], target: Sequence[AttributeSpec]) -> Tuple[float, Dict[str, Tuple[Optional[str], float]]]:
    if not source or not target:
        return 0.0, {}
    mapping: Dict[str, Tuple[Optional[str], float]] = {}
    scores = []
    for source_attr in source:
        best_name, best_score = None, 0.0
        for target_attr in target:
            lexical = lexical_similarity(source_attr.name, target_attr.name)
            type_score = type_similarity(source_attr.data_type, target_attr.data_type)
            semantic = cosine_similarity(source_attr.embedding, target_attr.embedding) if source_attr.embedding is not None and target_attr.embedding is not None else 0.0
            score = 0.55 * semantic + 0.30 * lexical + 0.15 * type_score if source_attr.embedding is not None and target_attr.embedding is not None else 0.7 * lexical + 0.3 * type_score
            if score > best_score:
                best_name, best_score = target_attr.name, score
        mapping[source_attr.name] = (best_name, best_score)
        scores.append(best_score)
    return float(np.mean(scores)), mapping


def structural_similarity(source: Sequence[AttributeSpec], target: Sequence[AttributeSpec]) -> float:
    if not source or not target:
        return 0.0
    count_ratio = min(len(source), len(target)) / max(len(source), len(target))
    keys = [attribute for attribute in source if attribute.is_key]
    if not keys:
        return count_ratio
    key_ratio = sum(any(lexical_similarity(key.name, column.name) > 0.5 for column in target) for key in keys) / len(keys)
    return 0.5 * count_ratio + 0.5 * key_ratio


class HybridReranker:
    def __init__(self, accept: float = 0.75, review: float = 0.55, min_margin: float = 0.08):
        self.accept = accept
        self.review = review
        self.min_margin = min_margin

    def score_pair(self, source: EntitySpec, target: EntitySpec) -> ScoreBreakdown:
        semantic = cosine_similarity(source.embedding, target.embedding) if source.embedding is not None and target.embedding is not None else 0.0
        lexical = lexical_similarity(source.name, target.name)
        columns, mapping = column_coverage(source.attributes, target.attributes)
        types = float(np.mean([type_similarity(next(a for a in source.attributes if a.name == name).data_type, next(a for a in target.attributes if a.name == target_name).data_type) for name, (target_name, _) in mapping.items() if target_name])) if mapping and any(target_name for target_name, _ in mapping.values()) else 0.5
        structure = structural_similarity(source.attributes, target.attributes)
        final = 0.45 * semantic + 0.15 * lexical + 0.25 * columns + 0.10 * types + 0.05 * structure
        return ScoreBreakdown(semantic, lexical, columns, types, structure, final)

    def match(self, source: EntitySpec, candidates: List[EntitySpec]) -> MatchResult:
        if not candidates:
            return MatchResult(source.name, None, 0.0, Decision.NO_MATCH, "No candidates retrieved.")
        scored = sorted(((candidate, self.score_pair(source, candidate)) for candidate in candidates), key=lambda item: item[1].final, reverse=True)
        best, breakdown = scored[0]
        second = scored[1][1].final if len(scored) > 1 else 0.0
        margin = breakdown.final - second
        if breakdown.final >= self.accept and margin >= self.min_margin:
            decision = Decision.MATCH
        elif breakdown.final >= self.review:
            decision = Decision.REVIEW
        else:
            decision = Decision.NO_MATCH
        target = best.name if decision is not Decision.NO_MATCH else None
        return MatchResult(source.name, target, round(breakdown.final, 4), decision, f"Hybrid score={breakdown.final:.3f}, margin={margin:.3f}.", breakdown, round(margin, 4), [(candidate.name, round(score.final, 4)) for candidate, score in scored[:5]], column_coverage(source.attributes, best.attributes)[1])
