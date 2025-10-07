from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.domain.entities.schema import ChangeType

@dataclass(frozen=True)
class SchemaChange:
    """Represents a single schema change operation."""
    change_type: ChangeType
    target_table: str
    target_column: Optional[str] = None
    definition: Optional[str] = None
    reason: str = ""
    sql: Optional[str] = None
    safe: bool = True
    requires_data_migration: bool = False
    estimated_impact: str = "low"  # low, medium, high


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
