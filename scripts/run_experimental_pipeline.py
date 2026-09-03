#!/usr/bin/env python3
"""
run_experimental_pipeline.py — STEP 3/PHASE 3-6 of the mission.

Orchestrates the ALREADY EXISTING classes end-to-end:

    U-Schema source elements
        -> AdvancedRAGRetriever (or a mock double in --dry-run/mock mode)
        -> CandidateResult[]                    (candidate_results_from_retrieval)
        -> SemanticDecisionService               (real)
        -> SemanticDecision
        -> MDEValidator                          (real)
        -> MDEValidationResult
        -> DecisionPolicy (uses RelevancePolicy) (real)
        -> DecisionResult
        -> EvolutionPlanner                      (real, for EVOLVE only)
        -> SchemaChange
        -> MigrationBuilder                      (real)
        -> SQL

This script does NOT reimplement any of the above. It only:
  1. loads a small controlled benchmark (data/experimental/open_world/),
  2. builds a mock AdvancedRAGRetriever + mock LLMOrchestrator whose
     canned answers come directly from the benchmark file (so the whole
     chain can be exercised deterministically, without a live vector
     store, embedding model, or LLM API key -- exactly what the mission
     requires: "Le système doit fonctionner sans LLM réel grâce à un mode
     mock/test"),
  3. calls the real domain services in the real order,
  4. writes artifacts/experimental/*.jsonl,
  5. prints the summary block.

--mode legacy is intentionally NOT faked: it prints an explicit
NOT_IMPLEMENTED notice rather than fabricating a result (mission rule:
"Ne fabrique PAS de labels arbitraires" / no fictional output).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.domain.entities.experimental import (  # noqa: E402
    DecisionType,
    MDEStatus,
    SemanticDecision,
)
from src.domain.entities.rag_schema import (  # noqa: E402
    FieldType,
    KnowledgeBaseDocument,
    SourceField,
)
from src.domain.entities.schema import DataType, USchemaAttribute  # noqa: E402
from src.domain.services.decision_policy import DecisionPolicy, DecisionThresholds  # noqa: E402
from src.domain.services.evolution_planner import EvolutionPlanner  # noqa: E402
from src.domain.services.mde_validator import MDEValidator, StructuralContext  # noqa: E402
from src.domain.services.migration_builder import MigrationBuilder  # noqa: E402
from src.domain.services.relevance_policy import RelevancePolicy  # noqa: E402
from src.infrastructure.rag.scoring_system import HybridScoringSystem  # noqa: E402
from src.infrastructure.rag.semantic_decision_service import SemanticDecisionService  # noqa: E402

DEFAULT_BENCHMARK = REPO_ROOT / "data/experimental/open_world/benchmark.jsonl"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/experimental"

_SOURCE_TYPE_TO_DATATYPE = {
    "string": DataType.STRING,
    "integer": DataType.INTEGER,
    "decimal": DataType.DECIMAL,
    "boolean": DataType.BOOLEAN,
    "timestamp": DataType.TIMESTAMP,
    "date": DataType.DATE,
    "json": DataType.JSON,
    "uuid": DataType.UUID,
}
_SOURCE_TYPE_TO_FIELDTYPE = {
    "string": FieldType.TEXT,
    "integer": FieldType.INTEGER,
    "decimal": FieldType.FLOAT,
    "boolean": FieldType.BOOLEAN,
    "timestamp": FieldType.DATETIME,
    "date": FieldType.DATETIME,
    "json": FieldType.TEXT,
    "uuid": FieldType.CODE,
}


# ---------------------------------------------------------------------------
# Mock doubles: canned answers coming straight from the benchmark file.
# Only these two I/O boundaries are mocked; every domain/service class
# downstream is the real one.
# ---------------------------------------------------------------------------


class BenchmarkRetriever:
    """Mock for AdvancedRAGRetriever.retrieve_candidates, sourced from the benchmark."""

    def __init__(self, candidates_spec: List[Dict[str, Any]]):
        self._candidates_spec = candidates_spec

    def retrieve_candidates(self, query):
        results = []
        for c in self._candidates_spec:
            table, column = c["target"].split(".", 1)
            doc = KnowledgeBaseDocument(
                id=c["target"],
                table=table,
                column=column,
                content=c["target"],
                metadata={
                    "data_type": c.get("target_data_type", "string"),
                    "constraints": {
                        "is_primary_key": c.get("is_primary_key", False),
                        "is_foreign_key": c.get("is_foreign_key", False),
                        "not_null": not c.get("nullable", True),
                    },
                    "description": c.get("rationale", ""),
                },
            )
            results.append((doc, c["embedding_score"], c["embedding_score"]))
        return results


class BenchmarkLLMOrchestrator:
    """Mock for LLMOrchestrator.match_field, sourced from the benchmark."""

    def __init__(self, candidates_spec: List[Dict[str, Any]]):
        self._candidates_spec = candidates_spec

    def match_field(self, query, docs, bi_scores, cross_scores):
        from types import SimpleNamespace

        candidates = [
            SimpleNamespace(
                target=c["target"],
                confidence_llm=c["llm_score"],
                confidence_model=c["embedding_score"],
                rationale=c.get("rationale", ""),
                guardrails=[],
            )
            for c in self._candidates_spec
        ]
        return SimpleNamespace(candidates=candidates)


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def source_field_from_item(item: Dict[str, Any]) -> SourceField:
    leaf = item["source_name"].split(".")[-1]
    return SourceField(
        path=item["source_name"],
        name_tokens=item["source_name"].split("."),
        inferred_type=_SOURCE_TYPE_TO_FIELDTYPE.get(item["source_type"], FieldType.TEXT),
        hints=[item.get("source_context", "")],
    )


def uschema_attribute_from_item(item: Dict[str, Any]) -> USchemaAttribute:
    leaf = item["source_name"].split(".")[-1]
    return USchemaAttribute(
        name=leaf,
        data_type=_SOURCE_TYPE_TO_DATATYPE.get(item["source_type"], DataType.STRING),
        required=False,
        is_key=(leaf == "id"),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_experimental(
    benchmark: List[Dict[str, Any]],
    top_k: int,
    threshold_match: float,
    threshold_review: float,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    scoring_system = HybridScoringSystem()
    mde_validator = MDEValidator()
    relevance_policy = RelevancePolicy()
    decision_policy = DecisionPolicy(
        thresholds=DecisionThresholds(tau_match=threshold_match, tau_review=threshold_review),
        relevance_policy=relevance_policy,
    )
    evolution_planner = EvolutionPlanner()
    migration_builder = MigrationBuilder()

    candidates_records: List[Dict[str, Any]] = []
    semantic_decisions_records: List[Dict[str, Any]] = []
    mde_records: List[Dict[str, Any]] = []
    decisions_records: List[Dict[str, Any]] = []
    evolution_records: List[Dict[str, Any]] = []
    all_schema_changes = []

    mde_status_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    change_type_counts: Dict[str, int] = {}

    for item in benchmark:
        source_id = item["source_id"]
        source_name = item["source_name"]
        source_field = source_field_from_item(item)
        source_attr = uschema_attribute_from_item(item)
        candidates_spec = item.get("candidates", [])

        retriever = BenchmarkRetriever(candidates_spec)
        llm_orchestrator = BenchmarkLLMOrchestrator(candidates_spec)
        service = SemanticDecisionService(
            retriever=retriever,
            llm_orchestrator=llm_orchestrator,
            scoring_system=scoring_system,
            top_k=top_k,
        )

        semantic_decisions: List[SemanticDecision] = service.evaluate_source_element(source_field)

        for sd in semantic_decisions:
            candidates_records.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "candidate_id": sd.candidate.target_element,
                    "rank": sd.candidate.rank,
                    "embedding_score": sd.embedding_score,
                    "llm_score": sd.llm_score,
                    "combined_score": sd.combined_score,
                }
            )
            semantic_decisions_records.append(
                {
                    "source_id": source_id,
                    "source_name": source_name,
                    "candidate": sd.candidate.target_element,
                    "embedding_score": sd.embedding_score,
                    "llm_score": sd.llm_score,
                    "combined_score": sd.combined_score,
                    "rationale": sd.rationale,
                }
            )

        best_semantic_decision: Optional[SemanticDecision] = None
        if semantic_decisions:
            best_semantic_decision = max(semantic_decisions, key=lambda d: d.combined_score)

        mde_result = None
        if best_semantic_decision is not None:
            best_candidate_spec = next(
                c for c in candidates_spec if c["target"] == best_semantic_decision.candidate.target_element
            )
            table, column = best_candidate_spec["target"].split(".", 1)
            structural_context = StructuralContext(
                is_primary_key=best_candidate_spec.get("is_primary_key", False),
                is_foreign_key=best_candidate_spec.get("is_foreign_key", False),
                nullable=best_candidate_spec.get("nullable", True),
                table_name=table,
                column_name=column,
            )
            mde_result = mde_validator.validate(
                source=source_attr,
                target_type=best_candidate_spec.get("target_data_type", "string"),
                structural_context=structural_context,
            )
            mde_records.append(
                {
                    "source_id": source_id,
                    "mde_status": mde_result.status.value,
                    "type_score": mde_result.type_score,
                    "structure_score": mde_result.structural_score,
                    "violations": mde_result.violations,
                }
            )
            mde_status_counts[mde_result.status.value] = mde_status_counts.get(mde_result.status.value, 0) + 1

        decision_result = decision_policy.decide(
            source_element=source_name,
            semantic_decision=best_semantic_decision,
            mde_validation=mde_result,
            semantic_evidence_for_relevance=item.get("semantic_evidence", 0.0),
        )
        decision_counts[decision_result.decision_type.value] = (
            decision_counts.get(decision_result.decision_type.value, 0) + 1
        )

        decision_record = {
            "source_id": source_id,
            "decision": decision_result.decision_type.value,
        }

        if decision_result.decision_type == DecisionType.EVOLVE:
            target_table = source_name.rsplit(".", 2)[0].rsplit(".", 1)[-1] if "." in source_name else None
            # For this benchmark's convention, the parent entity name is the
            # existing relational table (e.g. "patient.new_score" -> table "patient").
            parent = source_name.rsplit(".", 1)[0] if "." in source_name else None
            schema_change = evolution_planner.plan(
                decision=decision_result,
                source_attribute=source_attr,
                target_table=parent,
            )
            if schema_change is not None:
                all_schema_changes.append(schema_change)
                change_type_counts[schema_change.change_type.name] = (
                    change_type_counts.get(schema_change.change_type.name, 0) + 1
                )
                decision_record.update(
                    {
                        "change_type": schema_change.change_type.value,
                        "target_table": schema_change.target_table,
                        "target_column": schema_change.target_column,
                    }
                )
                evolution_records.append(
                    {
                        "source_id": source_id,
                        "change_type": schema_change.change_type.value,
                        "target_table": schema_change.target_table,
                        "target_column": schema_change.target_column,
                        "data_type": schema_change.data_type,
                        "predicted_operation": schema_change.change_type.value,
                        "gold_operation": item.get("expected_operation"),
                        "operation_match": schema_change.change_type.value == "add_column"
                        and item.get("expected_operation") == "ADD_COLUMN",
                    }
                )

        decisions_records.append(decision_record)

    sql_statements = migration_builder.build_migration(all_schema_changes) if all_schema_changes else []

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(artifacts_dir / "candidates.jsonl", candidates_records)
    _write_jsonl(artifacts_dir / "semantic_decisions.jsonl", semantic_decisions_records)
    _write_jsonl(artifacts_dir / "mde_validations.jsonl", mde_records)
    _write_jsonl(artifacts_dir / "decisions.jsonl", decisions_records)
    _write_jsonl(artifacts_dir / "evolution_operations.jsonl", evolution_records)
    _write_jsonl(artifacts_dir / "sql_migrations.jsonl", [{"sql": s} for s in sql_statements])

    summary = {
        "run_type": "controlled_mock",
        "source_elements": len(benchmark),
        "retrieved_candidates": len(candidates_records),
        "semantic_decisions": len(semantic_decisions_records),
        "mde_validations": mde_status_counts,
        "final_decisions": decision_counts,
        "evolution_operations": change_type_counts,
        "sql_statements_generated": len(sql_statements),
        "sql_statements": sql_statements,
    }
    _write_json(artifacts_dir / "summary.json", summary)

    return summary


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_summary(summary: Dict[str, Any], dry_run: bool) -> None:
    print("=" * 50)
    print("EXPERIMENTAL PIPELINE")
    print("=" * 50)
    print()
    print(f"Source elements: {summary['source_elements']}")
    print()
    print(f"Retrieved candidates: {summary['retrieved_candidates']}")
    print()
    print(f"Semantic decisions: {summary['semantic_decisions']}")
    print()
    print("MDE validations:")
    for status in ("VALID", "VALID_WITH_TRANSFORMATION", "REVIEW", "INVALID"):
        print(f"{status}: {summary['mde_validations'].get(status, 0)}")
    print()
    print("Final decisions:")
    for decision in ("MATCH", "EVOLVE", "REVIEW", "REJECT"):
        print(f"{decision}: {summary['final_decisions'].get(decision, 0)}")
    print()
    print("Evolution operations:")
    for op in ("ADD_TABLE", "CREATE_TABLE", "ADD_COLUMN", "ADD_FOREIGN_KEY", "ADD_ASSOCIATION_TABLE"):
        if op in summary["evolution_operations"] or op in ("ADD_TABLE", "ADD_COLUMN", "ADD_FOREIGN_KEY", "ADD_ASSOCIATION_TABLE"):
            print(f"{op}: {summary['evolution_operations'].get(op, 0)}")
    print()
    print(f"SQL statements generated: {summary['sql_statements_generated']}")
    print()
    print("Execution:")
    print("SKIPPED (dry-run)" if dry_run else "NOT_IMPLEMENTED (PostgreSQL executor not built yet)")
    print()
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experimental (or legacy) schema-evolution pipeline.")
    parser.add_argument("--mode", choices=["experimental", "legacy"], default="experimental")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold-match", type=float, default=0.85)
    parser.add_argument("--threshold-review", type=float, default=0.70)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    if args.mode == "legacy":
        print("=" * 50)
        print("LEGACY PIPELINE")
        print("=" * 50)
        print()
        print("NOT_IMPLEMENTED in this pass: legacy mode (RAGSchemaMatcher -> DiffEngine ->")
        print("MigrationBuilder) requires a populated RAGVectorStore / knowledge base, which")
        print("is out of scope for this dry-run controlled-benchmark exercise. See report.")
        return

    if not args.dry_run:
        print("Only --dry-run is implemented in this pass (no PostgreSQL executor yet).")
        print("Re-run with --dry-run.")
        sys.exit(1)

    def _relativize(p: Path) -> str:
        """Never write machine-specific absolute paths into artifacts."""
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    run_config = {
        "run_type": "controlled_mock",
        "run_type_note": (
            "Retrieval and LLM boundaries are mocked from a small hand-built "
            "benchmark (data/experimental/open_world/benchmark.jsonl). This is "
            "an integration-evidence run, NOT a scientific retrieval experiment. "
            "See scripts/reproduce_memory.py / artifacts/reproduction/ for the "
            "real, dataset-driven reproduction pipeline."
        ),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "top_k": args.top_k,
        "threshold_match": args.threshold_match,
        "threshold_review": args.threshold_review,
        "benchmark": _relativize(args.benchmark),
    }
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.artifacts_dir / "run_config.json", run_config)

    benchmark = load_benchmark(args.benchmark)
    summary = run_experimental(
        benchmark=benchmark,
        top_k=args.top_k,
        threshold_match=args.threshold_match,
        threshold_review=args.threshold_review,
        artifacts_dir=args.artifacts_dir,
    )
    print_summary(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()