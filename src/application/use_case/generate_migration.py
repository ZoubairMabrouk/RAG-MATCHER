"""Use case for generating migration plan."""
from typing import Dict, Any
from src.domain.repositeries.interfaces import ILLMClient
from src.domain.services.migration_builder import MigrationBuilder
from src.infrastructure.validators.sql_validator import SafetyValidator, SQLValidator
from src.domain.entities.evolution import EvolutionPlan

class GenerateMigrationUseCase:
    """
    Use case: Generate executable migration plan.
    Single Responsibility: Generate migration artifacts.
    """
    
    def __init__(
        self,
        llm_client: ILLMClient,
        migration_builder: MigrationBuilder,
        sql_validator: SQLValidator,
        safety_validator: SafetyValidator
    ):
        self._llm = llm_client
        self._migration_builder = migration_builder
        self._sql_validator = sql_validator
        self._safety_validator = safety_validator
    
    def execute(self, analysis_result: Dict[str, Any], use_llm: bool = True) -> EvolutionPlan:
        """Execute migration generation."""
        
        if use_llm:
            # Use LLM to generate sophisticated plan
            plan = self._llm.generate_evolution_plan(analysis_result)
        else:
            # Use deterministic builder
            changes = analysis_result["changes"]
            plan = EvolutionPlan(
                changes=changes,
                description="Automatically generated migration plan"
            )
        
        # Generate SQL
        sql_statements = self._migration_builder.build_migration(plan.changes)
        
        # Validate each statement
        validation_results = {}
        for i, sql in enumerate(sql_statements):
            is_valid, error = self._sql_validator.validate_syntax(sql)
            validation_results[f"statement_{i}"] = {
                "valid": is_valid,
                "error": error,
                "sql": sql
            }
        
        # Safety check
        all_safe = True
        safety_warnings = []
        for change in plan.changes:
            is_safe, warnings = self._sql_validator.validate_safety(change)
            if not is_safe:
                all_safe = False
            safety_warnings.extend(warnings)
        
        # Estimate duration
        duration = self._safety_validator.estimate_migration_duration(plan.changes)
        plan.estimated_duration_minutes = duration
        
        # Update validation results
        plan.validation_results = {
            "sql_valid": all(v["valid"] for v in validation_results.values()),
            "safe": all_safe,
            "warnings": safety_warnings
        }
        
        return plan
