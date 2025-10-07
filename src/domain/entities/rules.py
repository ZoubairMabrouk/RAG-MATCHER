from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.domain.entities.schema import DataType

@dataclass(frozen=True)
class NamingConvention:
    """Naming convention rules."""
    table_pattern: str = "snake_case"
    column_pattern: str = "snake_case"
    pk_suffix: str = "_id"
    fk_suffix: str = "_id"
    index_prefix: str = "idx_"
    constraint_prefix: str = "fk_"


@dataclass(frozen=True)
class DesignRule:
    """Represents a design rule (R1-R5 style)."""
    rule_id: str
    name: str
    description: str
    category: str  # naming, structure, performance, data_quality
    severity: str = "warning"  # info, warning, error, critical
    validation_query: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class RuleSet:
    """Collection of design rules."""
    naming: NamingConvention
    design_rules: List[DesignRule]
    type_mappings: Dict[DataType, str] = field(default_factory=dict)
    scd_policies: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
