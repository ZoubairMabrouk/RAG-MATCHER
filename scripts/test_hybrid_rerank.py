"""
hybrid_reranker.py

Schema-aware hybrid reranker for the U-Schema <-> relational-schema matching
pipeline (BioBERT + FAISS retrieval feeding into a calibrated decision).

Replaces "highest cosine wins" with:

    score = w_sem    * S_semantic
          + w_lex    * S_lexical
          + w_col    * S_columns
          + w_type   * S_types
          + w_struct * S_structure

then applies a MATCH / REVIEW / NO_MATCH decision using both an absolute
threshold and a margin-over-second-best check, so that near-ties (e.g.
patients=0.81 vs profiles=0.80) fall back to REVIEW instead of a false MATCH.

This module has no hard dependency on the rest of the codebase: it works
with plain (name, data_type, is_key, embedding) tuples wrapped in
AttributeSpec/EntitySpec, so it can be dropped next to
`embedding_service.py` and wired into `RAGSchemaMatcher` without changing
domain entities. See `example_wire_into_matcher()` at the bottom for the
integration point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import inflect
    _INFLECT = inflect.engine()
except Exception:  # optional dependency
    _INFLECT = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttributeSpec:
    """A single U-Schema attribute or relational column, normalized."""
    name: str
    data_type: str = "string"
    is_key: bool = False
    embedding: Optional[np.ndarray] = None  # optional attribute-level vector


@dataclass
class EntitySpec:
    """A U-Schema entity (query side) or a relational table (candidate side)."""
    name: str
    attributes: List[AttributeSpec] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # entity/table-level vector


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


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def normalize_path(name: str) -> str:
    """
    Flatten JSON-nested / relation-style names before comparison.

        "address->address"             -> "address address"
        "ai_annotations->aiannotation" -> "ai annotations ai annotation"
    """
    if not name:
        return ""
    name = name.replace("->", " ")
    name = name.replace("_", " ")
    name = _CAMEL_RE.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def tokenize(name: str) -> List[str]:
    return [t for t in normalize_path(name).split(" ") if t]


def singularize(token: str) -> str:
    if _INFLECT is not None:
        singular = _INFLECT.singular_noun(token)
        return singular if singular else token
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalized_tokens(name: str) -> List[str]:
    return [singularize(t) for t in tokenize(name)]


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

def cosine_similarity(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and N candidate vectors."""
    query = np.asarray(query_embedding, dtype=np.float32)
    candidates = np.atleast_2d(np.asarray(candidate_embeddings, dtype=np.float32))

    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(len(candidates), dtype=np.float32)
    query = query / q_norm

    norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    candidates = candidates / norms

    return candidates @ query


def lexical_similarity(a: str, b: str) -> float:
    """
    Name similarity blending token overlap (handles singular/plural,
    snake_case, camelCase) with a character-level ratio.
    """
    if not a or not b:
        return 0.0

    a_norm, b_norm = normalize_path(a), normalize_path(b)
    if a_norm == b_norm:
        return 1.0

    a_tokens, b_tokens = set(normalized_tokens(a)), set(normalized_tokens(b))
    token_overlap = (
        len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
        if a_tokens and b_tokens else 0.0
    )
    char_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()

    return 0.6 * token_overlap + 0.4 * char_ratio


_TYPE_FAMILIES: Dict[str, set] = {
    "string": {"string", "varchar", "text", "char", "nvarchar", "character varying"},
    "integer": {"integer", "int", "bigint", "smallint", "serial", "bigserial"},
    "float": {"float", "decimal", "double", "real", "numeric"},
    "boolean": {"boolean", "bool"},
    "date": {"date"},
    "timestamp": {"timestamp", "datetime", "timestamptz"},
    "json": {"json", "jsonb"},
    "uuid": {"uuid"},
}

# Cross-family pairs that are "close enough" to avoid a hard 0.
_SOFT_COMPAT: Dict[Tuple[str, str], float] = {
    ("date", "timestamp"): 0.85,
    ("timestamp", "date"): 0.85,
    ("integer", "float"): 0.6,
    ("float", "integer"): 0.6,
    ("string", "json"): 0.3,
    ("json", "string"): 0.3,
    ("string", "uuid"): 0.4,
    ("uuid", "string"): 0.4,
}


def _canonical_family(dtype: str) -> Optional[str]:
    d = (dtype or "").strip().lower()
    for family, members in _TYPE_FAMILIES.items():
        if d == family or d in members:
            return family
    return None


def type_similarity(source_type: str, target_type: str) -> float:
    if not source_type or not target_type:
        return 0.5  # unknown -> neutral rather than punishing
    s, t = source_type.strip().lower(), target_type.strip().lower()
    if s == t:
        return 1.0
    fs, ft = _canonical_family(s), _canonical_family(t)
    if fs is None or ft is None:
        return 0.5
    if fs == ft:
        return 1.0
    return _SOFT_COMPAT.get((fs, ft), 0.1)


def column_coverage(
    source_attrs: Sequence[AttributeSpec],
    target_attrs: Sequence[AttributeSpec],
) -> Tuple[float, Dict[str, Tuple[Optional[str], float]]]:
    """
    For each source attribute, find its best-matching target column via a
    blend of (optional) attribute-level embedding similarity, lexical
    similarity, and type compatibility. Average the best-per-attribute
    scores into a single coverage score.

    Returns (coverage_score, mapping) where
        mapping[attr_name] = (best_matching_column_name_or_None, score)
    """
    if not source_attrs or not target_attrs:
        return 0.0, {}

    mapping: Dict[str, Tuple[Optional[str], float]] = {}
    best_scores: List[float] = []

    for sa in source_attrs:
        best_name, best_score = None, 0.0
        for ta in target_attrs:
            lex = lexical_similarity(sa.name, ta.name)
            typ = type_similarity(sa.data_type, ta.data_type)

            if sa.embedding is not None and ta.embedding is not None:
                sem = float(cosine_similarity(sa.embedding, ta.embedding[np.newaxis, :])[0])
                combined = 0.55 * sem + 0.30 * lex + 0.15 * typ
            else:
                combined = 0.7 * lex + 0.3 * typ

            if combined > best_score:
                best_name, best_score = ta.name, combined

        mapping[sa.name] = (best_name, best_score)
        best_scores.append(best_score)

    return float(np.mean(best_scores)), mapping


def structural_similarity(
    source_attrs: Sequence[AttributeSpec],
    target_attrs: Sequence[AttributeSpec],
) -> float:
    """
    Cheap structural signal: how close attribute/column counts are, plus
    whether key attributes have a plausible lexical counterpart. Penalizes
    matching a rich entity to a near-empty table (or vice versa).
    """
    n_s, n_t = len(source_attrs), len(target_attrs)
    if n_s == 0 or n_t == 0:
        return 0.0
    count_ratio = min(n_s, n_t) / max(n_s, n_t)

    keys = [a for a in source_attrs if a.is_key]
    if not keys:
        return count_ratio

    key_hits = sum(
        1 for k in keys
        if any(lexical_similarity(k.name, t.name) > 0.5 for t in target_attrs)
    )
    key_ratio = key_hits / len(keys)

    return 0.5 * count_ratio + 0.5 * key_ratio


# ---------------------------------------------------------------------------
# Hybrid scorer
# ---------------------------------------------------------------------------

@dataclass
class HybridWeights:
    semantic: float = 0.45
    lexical: float = 0.15
    columns: float = 0.25
    types: float = 0.10
    structure: float = 0.05

    def normalized(self) -> "HybridWeights":
        total = self.semantic + self.lexical + self.columns + self.types + self.structure
        if total == 0:
            return self
        return HybridWeights(
            semantic=self.semantic / total,
            lexical=self.lexical / total,
            columns=self.columns / total,
            types=self.types / total,
            structure=self.structure / total,
        )


@dataclass
class DecisionThresholds:
    accept: float = 0.75
    review: float = 0.55
    min_margin: float = 0.08  # best-vs-second-best gap required for MATCH


class HybridReranker:
    """
    Combines semantic retrieval (from BioBERT/FAISS) with lexical, column,
    type, and structural signals to rerank candidate tables/entities and
    produce a calibrated MATCH / REVIEW / NO_MATCH decision.

    Typical usage inside RAGSchemaMatcher.match_table():

        reranker = HybridReranker()
        result = reranker.match(source_entity, candidate_entities)
    """

    def __init__(
        self,
        weights: Optional[HybridWeights] = None,
        thresholds: Optional[DecisionThresholds] = None,
    ):
        self.weights = (weights or HybridWeights()).normalized()
        self.thresholds = thresholds or DecisionThresholds()

    def score_pair(self, source: EntitySpec, target: EntitySpec) -> ScoreBreakdown:
        if source.embedding is not None and target.embedding is not None:
            s_sem = float(cosine_similarity(source.embedding, target.embedding[np.newaxis, :])[0])
        else:
            s_sem = 0.0

        s_lex = lexical_similarity(source.name, target.name)
        s_col, _ = column_coverage(source.attributes, target.attributes)
        s_struct = structural_similarity(source.attributes, target.attributes)

        # Aggregate type compatibility across the best-matched columns.
        _, mapping = column_coverage(source.attributes, target.attributes)
        type_scores = []
        by_name = {a.name: a for a in target.attributes}
        for sa in source.attributes:
            best_name, _ = mapping.get(sa.name, (None, 0.0))
            if best_name and best_name in by_name:
                type_scores.append(type_similarity(sa.data_type, by_name[best_name].data_type))
        s_type = float(np.mean(type_scores)) if type_scores else 0.5

        w = self.weights
        final = (
            w.semantic * s_sem
            + w.lexical * s_lex
            + w.columns * s_col
            + w.types * s_type
            + w.structure * s_struct
        )

        return ScoreBreakdown(
            semantic=s_sem, lexical=s_lex, columns=s_col,
            types=s_type, structure=s_struct, final=final,
        )

    def match(self, source: EntitySpec, candidates: List[EntitySpec]) -> MatchResult:
        if not candidates:
            return MatchResult(
                source_name=source.name, target_name=None, confidence=0.0,
                decision=Decision.NO_MATCH, rationale="No candidates retrieved.",
            )

        scored = [(c, self.score_pair(source, c)) for c in candidates]
        scored.sort(key=lambda x: x[1].final, reverse=True)

        best_candidate, best_breakdown = scored[0]
        second_score = scored[1][1].final if len(scored) > 1 else 0.0
        margin = best_breakdown.final - second_score

        decision, rationale = self._decide(best_breakdown.final, margin)

        _, column_mapping = column_coverage(source.attributes, best_candidate.attributes)

        return MatchResult(
            source_name=source.name,
            target_name=best_candidate.name if decision != Decision.NO_MATCH else None,
            confidence=round(best_breakdown.final, 4),
            decision=decision,
            rationale=rationale,
            breakdown=best_breakdown,
            margin=round(margin, 4),
            top_candidates=[(c.name, round(b.final, 4)) for c, b in scored[:5]],
            column_mapping=column_mapping,
        )

    def _decide(self, score: float, margin: float) -> Tuple[Decision, str]:
        t = self.thresholds
        if score >= t.accept and margin >= t.min_margin:
            return Decision.MATCH, (
                f"Score {score:.3f} >= accept threshold {t.accept} "
                f"with margin {margin:.3f} over runner-up."
            )
        if score >= t.accept and margin < t.min_margin:
            return Decision.REVIEW, (
                f"Score {score:.3f} clears the accept threshold but margin "
                f"{margin:.3f} < {t.min_margin}: too close to the runner-up to trust."
            )
        if score >= t.review:
            return Decision.REVIEW, f"Score {score:.3f} is in the review band [{t.review}, {t.accept})."
        return Decision.NO_MATCH, f"Score {score:.3f} is below the review threshold {t.review}."


# ---------------------------------------------------------------------------
# Integration sketch (not executed) — how this plugs into RAGSchemaMatcher
# ---------------------------------------------------------------------------

def example_wire_into_matcher():
    """
    Sketch of how to wire HybridReranker into the existing pipeline described
    in embedding_service.py / test_rag_generation.py, replacing
    "highest cosine wins" with the hybrid decision.

    Inside RAGSchemaMatcher.match_table(entity_name, attr_names, ...):

        # 1) existing retrieval step (unchanged): BioBERT + FAISS
        query_emb = self.embedding_service.embed_query(query)
        candidate_docs, candidate_embs = self.vector_store.search(
            query_emb, top_k=self.top_k_search
        )

        # 2) build EntitySpec objects for source + candidates
        source = EntitySpec(
            name=entity_name,
            embedding=query_emb,
            attributes=[
                AttributeSpec(name=a.name, data_type=a.data_type, is_key=a.is_key)
                for a in entity.attributes
            ],
        )
        candidates = [
            EntitySpec(
                name=doc.table_name,
                embedding=emb,
                attributes=[
                    AttributeSpec(name=c.name, data_type=c.data_type, is_key=c.is_key)
                    for c in doc.columns
                ],
            )
            for doc, emb in zip(candidate_docs, candidate_embs)
        ]

        # 3) hybrid rerank + calibrated decision (replaces raw cosine argmax)
        reranker = HybridReranker()  # or inject a shared instance
        result = reranker.match(source, candidates)

        return TableMatchResult(
            target_name=result.target_name,
            confidence=result.confidence,
            rationale=result.rationale,
        )

    For match_column(), call `column_coverage()` directly on a single
    (source_attrs=[the one attribute], target_attrs=[table.columns]) pair,
    or reuse `result.column_mapping` already computed by match_table().
    """
    raise NotImplementedError("This function is documentation, not runnable code.")


if __name__ == "__main__":
    # Minimal smoke test without any embeddings (lexical/type/structure only)
    # — mirrors the bad cases from the logs (contact -> purchases, etc.)
    # to show the hybrid score correctly demotes them even without vectors.
    patient = EntitySpec(
        name="patient",
        attributes=[
            AttributeSpec("patient_id", "integer", is_key=True),
            AttributeSpec("gender", "string"),
            AttributeSpec("dob", "date"),
            AttributeSpec("deceased", "boolean"),
            AttributeSpec("allergies", "string"),
        ],
    )
    patients_table = EntitySpec(
        name="patients",
        attributes=[
            AttributeSpec("subject_id", "integer", is_key=True),
            AttributeSpec("gender", "varchar"),
            AttributeSpec("dob", "date"),
            AttributeSpec("dod", "date"),
            AttributeSpec("expire_flag", "boolean"),
        ],
    )
    notifications_table = EntitySpec(
        name="notifications",
        attributes=[
            AttributeSpec("notification_id", "integer", is_key=True),
            AttributeSpec("message", "text"),
            AttributeSpec("sent_at", "timestamp"),
            AttributeSpec("read_flag", "boolean"),
        ],
    )

    reranker = HybridReranker()
    result = reranker.match(patient, [patients_table, notifications_table])

    print(f"{result.source_name} -> {result.target_name} "
          f"(confidence={result.confidence}, decision={result.decision.value})")
    print(f"rationale: {result.rationale}")
    print(f"top candidates: {result.top_candidates}")
    print(f"column mapping: {result.column_mapping}")