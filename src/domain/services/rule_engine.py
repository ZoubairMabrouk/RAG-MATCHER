from typing import List, Dict, Optional
from src.domain.entities.evolution import ChangeType, SchemaChange
from src.domain.entities.rules import RuleSet, DesignRule

class RuleEngine:
    """
    Validates schema changes against design rules.
    Single Responsibility: Rule validation only.
    """
    
    def __init__(self, rule_set: RuleSet):
        self._rules = rule_set
    
    def validate_changes(self, changes: List[SchemaChange]) -> Dict[str, List[str]]:
        """
        Validate changes against design rules.
        Returns dict of violations by rule_id.
        """
        violations = {}
        
        for change in changes:
            for rule in self._rules.design_rules:
                violation = self._check_rule(change, rule)
                if violation:
                    if rule.rule_id not in violations:
                        violations[rule.rule_id] = []
                    violations[rule.rule_id].append(violation)
        
        return violations
    
    def _check_rule(self, change: SchemaChange, rule: DesignRule) -> Optional[str]:
        """Check if a change violates a specific rule."""
        # Example rule checks
        if rule.rule_id == "R1_NAMING":
            return self._check_naming_convention(change)
        elif rule.rule_id == "R2_NO_NULL_FK":
            return self._check_nullable_fk(change)
        elif rule.rule_id == "R3_INDEX_FK":
            return self._check_fk_index(change)
        
        return None
    
    def _check_naming_convention(self, change: SchemaChange) -> Optional[str]:
        """Validate naming conventions."""
        if change.target_table:
            if not change.target_table.islower():
                return f"Table '{change.target_table}' does not follow snake_case"
            if '_' not in change.target_table and len(change.target_table) > 8:
                return f"Table '{change.target_table}' lacks underscores"
        
        return None
    
    def _check_nullable_fk(self, change: SchemaChange) -> Optional[str]:
        """Check foreign key nullability."""
        if change.change_type == ChangeType.ADD_CONSTRAINT:
            if change.definition and "FOREIGN KEY" in change.definition:
                if "NOT NULL" not in str(change.definition):
                    return f"Foreign key '{change.target_column}' should be NOT NULL"
        
        return None
    
    def _check_fk_index(self, change: SchemaChange) -> Optional[str]:
        """Ensure foreign keys have indexes."""
        # This would require cross-referencing with index changes
        return None
    
    def enrich_changes_with_best_practices(self, changes: List[SchemaChange]) -> List[SchemaChange]:
        """Add recommended changes based on best practices."""
        enriched = list(changes)
        
        for change in changes:
            if change.change_type == ChangeType.ADD_CONSTRAINT:
                # Suggest index for FK
                if change.target_column:
                    enriched.append(SchemaChange(
                        change_type=ChangeType.ADD_INDEX,
                        target_table=change.target_table,
                        target_column=change.target_column,
                        definition=f"INDEX ON {change.target_table}({change.target_column})",
                        reason="Best practice: Index foreign key columns",
                        safe=True,
                        estimated_impact="low"
                    ))
        
        return enriched
