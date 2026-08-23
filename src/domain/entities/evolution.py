from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.domain.entities.schema import ChangeType

@dataclass(frozen=True)
class SchemaChange:
    """Represents a single schema change operation.

    STEP 7 extension (EvolutionPlanner, see docs/experimental_architecture.md
    section 5): fields below were added so that SchemaChange can directly play
    the role the brief calls "EvolutionOperation" instead of introducing a
    second, competing taxonomy. All new fields are optional/defaulted so every
    existing SchemaChange(...) call site keeps working unmodified.
    """
    change_type: ChangeType
    target_table: str
    target_column: Optional[str] = None
    definition: Optional[str] = None
    reason: str = ""
    sql: Optional[str] = None
    safe: bool = True
    requires_data_migration: bool = False
    estimated_impact: str = "low"  # low, medium, high
    # --- evolution-operation metadata (new, additive) ---
    data_type: Optional[str] = None
    primary_key: bool = False
    foreign_key: Optional[str] = None  # "referenced_table.referenced_column"
    cardinality: Optional[str] = None  # e.g. "one_to_many", "many_to_many"
    dependencies: List[str] = field(default_factory=list)
    rationale: Optional[str] = None
    source_element: Optional[str] = None  # traceability back to the NoSQL source path


@dataclass
class EvolutionPlan:
    """Complete plan for schema evolution."""
    changes: List[SchemaChange]
    description: str
    risk_level: str = "low"  # low, medium, high, critical
    estimated_duration_minutes: int = 0
    backward_compatible: bool = True
    rollback_plan: Optional[str] = None
    validation_results: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)