"""
MDEValidator — STEP 5 (main new contribution of this phase).

Responsibility: decide whether a candidate relational target is structurally
and type-compatible with a source U-Schema attribute, independently of any
LLM judgment. This is deliberately pure/deterministic so it can be unit
tested without a network call, an embedding model, or an LLM.

Per the mission's explicit rule:
    "LLM says MATCH + MDE says INVALID must NOT produce MATCH."
MDEValidator is the structural authority; DecisionPolicy (STEP 6) is the one
that combines this with the semantic decision, but MDEValidator itself does
not know about LLM scores at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.domain.entities.experimental import MDEStatus, MDEValidationResult
from src.domain.entities.schema import DataType, USchemaAttribute


# ---------------------------------------------------------------------------
# Type compatibility table
# ---------------------------------------------------------------------------
# Maps (source_type, target_type) -> ("compatible" | "compatible_with_transformation" | "incompatible")
# Types are normalized lowercase strings so both DataType enum values and raw
# SQL type strings (e.g. "VARCHAR(255)", "TIMESTAMP") coming from the
# introspected relational schema can be compared.

_COMPATIBLE = "compatible"
_COMPATIBLE_WITH_TRANSFORM = "compatible_with_transformation"
_INCOMPATIBLE = "incompatible"

_TYPE_ALIASES = {
    # normalize a variety of raw SQL / DataType spellings to a small canonical set
    "integer": "integer", "int": "integer", "bigint": "integer", "smallint": "integer",
    "decimal": "decimal", "numeric": "decimal", "float": "decimal", "double": "decimal", "real": "decimal",
    "string": "string", "varchar": "string", "text": "string", "char": "string",
    "boolean": "boolean", "bool": "boolean", "bit": "boolean",
    "timestamp": "timestamp", "datetime": "timestamp",
    "date": "date",
    "json": "json", "jsonb": "json",
    "uuid": "uuid",
}


_CANONICAL_TYPES = {"integer", "decimal", "string", "boolean", "timestamp", "date", "json", "uuid"}


def _normalize_type(raw: str) -> str:
    if raw is None:
        return "unknown"
    key = str(raw).strip().lower()
    # strip length/precision annotations, e.g. "varchar(255)" -> "varchar"
    if "(" in key:
        key = key.split("(", 1)[0].strip()
    canonical = _TYPE_ALIASES.get(key, key)
    if canonical not in _CANONICAL_TYPES:
        return "unknown"
    return canonical


# canonical_source -> canonical_target -> compatibility label
_TYPE_TABLE: Dict[str, Dict[str, str]] = {
    "integer": {
        "integer": _COMPATIBLE,
        "decimal": _COMPATIBLE_WITH_TRANSFORM,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "boolean": _INCOMPATIBLE,
        "timestamp": _INCOMPATIBLE,
        "date": _INCOMPATIBLE,
        "json": _INCOMPATIBLE,
        "uuid": _INCOMPATIBLE,
    },
    "decimal": {
        "integer": _COMPATIBLE_WITH_TRANSFORM,
        "decimal": _COMPATIBLE,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "boolean": _INCOMPATIBLE,
        "timestamp": _INCOMPATIBLE,
        "date": _INCOMPATIBLE,
        "json": _INCOMPATIBLE,
        "uuid": _INCOMPATIBLE,
    },
    "string": {
        "string": _COMPATIBLE,
        "integer": _COMPATIBLE_WITH_TRANSFORM,
        "decimal": _COMPATIBLE_WITH_TRANSFORM,
        "boolean": _COMPATIBLE_WITH_TRANSFORM,
        "date": _INCOMPATIBLE,
        "timestamp": _INCOMPATIBLE,
        "json": _COMPATIBLE_WITH_TRANSFORM,
        "uuid": _COMPATIBLE_WITH_TRANSFORM,
    },
    "boolean": {
        "boolean": _COMPATIBLE,
        "integer": _COMPATIBLE_WITH_TRANSFORM,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "decimal": _INCOMPATIBLE,
        "date": _INCOMPATIBLE,
        "timestamp": _INCOMPATIBLE,
        "json": _INCOMPATIBLE,
        "uuid": _INCOMPATIBLE,
    },
    "timestamp": {
        "timestamp": _COMPATIBLE,
        "date": _COMPATIBLE_WITH_TRANSFORM,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "integer": _INCOMPATIBLE,
        "decimal": _INCOMPATIBLE,
        "boolean": _INCOMPATIBLE,
        "json": _INCOMPATIBLE,
        "uuid": _INCOMPATIBLE,
    },
    "date": {
        "date": _COMPATIBLE,
        "timestamp": _COMPATIBLE_WITH_TRANSFORM,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "integer": _INCOMPATIBLE,
        "decimal": _INCOMPATIBLE,
        "boolean": _INCOMPATIBLE,
        "json": _INCOMPATIBLE,
        "uuid": _INCOMPATIBLE,
    },
    "json": {
        "json": _COMPATIBLE,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "integer": _INCOMPATIBLE,
        "decimal": _INCOMPATIBLE,
        "boolean": _INCOMPATIBLE,
        "date": _INCOMPATIBLE,
        "timestamp": _INCOMPATIBLE,
        "uuid": _INCOMPATIBLE,
    },
    "uuid": {
        "uuid": _COMPATIBLE,
        "string": _COMPATIBLE_WITH_TRANSFORM,
        "integer": _INCOMPATIBLE,
        "decimal": _INCOMPATIBLE,
        "boolean": _INCOMPATIBLE,
        "date": _INCOMPATIBLE,
        "timestamp": _INCOMPATIBLE,
        "json": _INCOMPATIBLE,
    },
}

_TYPE_SCORE = {
    _COMPATIBLE: 1.0,
    _COMPATIBLE_WITH_TRANSFORM: 0.6,
    _INCOMPATIBLE: 0.0,
}


@dataclass(frozen=True)
class StructuralContext:
    """
    Minimal structural facts about the CANDIDATE (relational) side needed for
    validation, gathered from Table/Column metadata already present in the
    project (see rag_schema_matcher._create_column_document /
    src.domain.entities.schema.Column/Table).
    """
    is_primary_key: bool = False
    is_foreign_key: bool = False
    nullable: bool = True
    table_name: Optional[str] = None
    column_name: Optional[str] = None


class MDEValidator:
    """
    Single Responsibility: structural + type validation of a candidate match.
    Does NOT do retrieval, does NOT call an LLM, does NOT decide MATCH/EVOLVE/
    REVIEW/REJECT (that's DecisionPolicy) -- it only produces the evidence
    DecisionPolicy needs.
    """

    def __init__(
        self,
        review_on_incompatible_type: bool = False,
        review_on_key_mismatch: bool = True,
    ):
        """
        Args:
            review_on_incompatible_type: if True, an incompatible type
                produces MDEStatus.REVIEW instead of INVALID (policy choice,
                see brief STEP 5.4 test #3: "INVALID ou REVIEW selon la
                politique" -- default is INVALID, which is the stricter,
                more defensible default for a first release).
            review_on_key_mismatch: if True, a foreign-key mismatch (source
                looks like it should reference a PK but candidate isn't one,
                or vice versa) produces REVIEW rather than INVALID.
        """
        self._review_on_incompatible_type = review_on_incompatible_type
        self._review_on_key_mismatch = review_on_key_mismatch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        source: USchemaAttribute,
        target_type: str,
        structural_context: Optional[StructuralContext] = None,
    ) -> MDEValidationResult:
        """
        Validate a source attribute against a candidate target column.

        Args:
            source: the U-Schema attribute being matched.
            target_type: raw or normalized SQL type of the candidate column
                (e.g. "INTEGER", "VARCHAR(255)", DataType.STRING.value).
            structural_context: structural facts about the candidate column;
                if omitted, structural/key/cardinality checks are skipped
                (score 0.5, neutral) rather than penalized -- we do not want
                to fabricate a violation with no evidence.
        """
        violations = []

        type_label, type_score = self._check_type(source.data_type, target_type)
        if type_label == _INCOMPATIBLE:
            violations.append(
                f"type_incompatible: {source.data_type.value} !~ {_normalize_type(target_type)}"
            )

        structural_score, structural_violations = self._check_structure(source, structural_context)
        violations.extend(structural_violations)

        key_score, key_violations = self._check_keys(source, structural_context)
        violations.extend(key_violations)

        cardinality_score, cardinality_violations = self._check_cardinality(source, structural_context)
        violations.extend(cardinality_violations)

        status = self._resolve_status(type_label, structural_score, key_score, violations)

        explanation = self._build_explanation(type_label, structural_score, key_score, violations)

        return MDEValidationResult(
            status=status,
            type_score=type_score,
            structural_score=structural_score,
            key_score=key_score,
            cardinality_score=cardinality_score,
            violations=violations,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # STEP 5.1 — type validation
    # ------------------------------------------------------------------

    def _check_type(self, source_type: DataType, target_type: str) -> Tuple[str, float]:
        src = _normalize_type(source_type.value if isinstance(source_type, DataType) else source_type)
        tgt = _normalize_type(target_type)

        if src == "unknown" or tgt == "unknown":
            return _COMPATIBLE_WITH_TRANSFORM, 0.5  # not enough evidence to penalize

        row = _TYPE_TABLE.get(src, {})
        label = row.get(tgt, _INCOMPATIBLE)
        return label, _TYPE_SCORE[label]

    # ------------------------------------------------------------------
    # STEP 5.2 / 5.3 — object/array/scalar + structural validation
    # ------------------------------------------------------------------

    def _check_structure(
        self, source: USchemaAttribute, ctx: Optional[StructuralContext]
    ) -> Tuple[float, list]:
        violations = []

        # A source that is an object or array can never map directly to a
        # scalar relational column -- this is the "patient.device" vs
        # "device_id" example from the brief (§7 / STEP 5.2).
        if source.is_object:
            violations.append(
                f"object_to_scalar: '{'.'.join(source.nesting_path) or source.name}' "
                "is a nested object, not a leaf value; cannot map directly to a scalar column"
            )
            return 0.0, violations

        if source.is_array:
            violations.append(
                f"array_to_scalar: '{'.'.join(source.nesting_path) or source.name}' "
                "is an array; cannot map directly to a scalar column without an explicit "
                "aggregation/child-table strategy"
            )
            return 0.0, violations

        # Leaf scalar attribute -- no structural evidence against it either
        # way. Absence of a StructuralContext means "no evidence", which we
        # treat as a pass (1.0), not a penalty: we do not fabricate a
        # violation with no evidence, per this validator's stated policy.
        return 1.0, violations

    # ------------------------------------------------------------------
    # key role validation
    # ------------------------------------------------------------------

    def _check_keys(
        self, source: USchemaAttribute, ctx: Optional[StructuralContext]
    ) -> Tuple[float, list]:
        violations = []
        if ctx is None:
            return 1.0, violations

        if source.is_key and not (ctx.is_primary_key or ctx.is_foreign_key):
            violations.append(
                f"key_role_mismatch: source '{source.name}' is flagged as a key "
                f"but candidate column '{ctx.column_name}' is neither PK nor FK"
            )
            return 0.2, violations

        if not source.is_key and ctx.is_primary_key:
            violations.append(
                f"key_role_mismatch: candidate column '{ctx.column_name}' is a primary key "
                f"but source '{source.name}' is not flagged as a key"
            )
            return 0.4, violations

        return 1.0, violations

    # ------------------------------------------------------------------
    # cardinality validation (best-effort; no dedicated cardinality metadata
    # exists yet on USchemaAttribute beyond is_array, so this stays simple
    # and honest about what it can check)
    # ------------------------------------------------------------------

    def _check_cardinality(
        self, source: USchemaAttribute, ctx: Optional[StructuralContext]
    ) -> Tuple[float, list]:
        violations = []
        if source.is_array and ctx is not None and not ctx.is_foreign_key:
            violations.append(
                "cardinality_mismatch: array-valued source usually implies a one-to-many "
                "relationship, which requires a child/association table, not a plain column"
            )
            return 0.2, violations
        return 1.0, violations

    # ------------------------------------------------------------------
    # status resolution
    # ------------------------------------------------------------------

    def _resolve_status(
        self, type_label: str, structural_score: float, key_score: float, violations: list
    ) -> MDEStatus:
        # Hard structural violations (object/array to scalar) always win: no
        # semantic score can override a structural impossibility.
        if structural_score == 0.0:
            return MDEStatus.INVALID

        if type_label == _INCOMPATIBLE:
            return MDEStatus.REVIEW if self._review_on_incompatible_type else MDEStatus.INVALID

        if key_score <= 0.2:
            return MDEStatus.REVIEW if self._review_on_key_mismatch else MDEStatus.INVALID

        if type_label == _COMPATIBLE_WITH_TRANSFORM:
            return MDEStatus.VALID_WITH_TRANSFORMATION

        if key_score < 1.0 or structural_score < 1.0:
            return MDEStatus.REVIEW

        return MDEStatus.VALID

    def _build_explanation(
        self, type_label: str, structural_score: float, key_score: float, violations: list
    ) -> str:
        if not violations:
            return "Type, structure, and key role are all compatible."
        return "; ".join(violations)