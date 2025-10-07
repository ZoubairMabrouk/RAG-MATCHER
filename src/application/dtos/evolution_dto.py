"""Data Transfer Objects for application layer."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from src.domain.entities.evolution import EvolutionPlan
@dataclass
class EvolutionRequest:
    """Request to generate evolution plan."""
    uschema_json: Dict[str, Any]
    database_connection: str
    include_rag: bool = True
    dry_run: bool = True
    safe_mode: bool = True
    target_dialect: str = "postgresql"


@dataclass
class EvolutionResponse:
    """Response containing evolution plan."""
    plan: EvolutionPlan
    sql_statements: List[str]
    validation_results: Dict[str, Any]
    execution_report: Optional[str] = None
    estimated_duration_minutes: int = 0