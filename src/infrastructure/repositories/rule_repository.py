from typing import Optional,List, Any, Dict
from src.domain.repositeries.interfaces import IRuleRepository

from src.domain.entities.rules import RuleSet, DesignRule,DataType, NamingConvention
class RuleRepository(IRuleRepository):
    """
    Repository for design rules.
    Single Responsibility: Rule data access.
    """
    
    def __init__(self, rules_file: Optional[str] = None):
        self._rules_file = rules_file
        self._default_rules = self._create_default_rules()
    
    def get_all_rules(self) -> RuleSet:
        """Retrieve all design rules."""
        if self._rules_file:
            # Load from file
            import json
            with open(self._rules_file, 'r') as f:
                data = json.load(f)
            # Parse rules from JSON
            return self._parse_rules(data)
        
        return self._default_rules
    
    def get_rules_by_category(self, category: str) -> List[DesignRule]:
        """Retrieve rules by category."""
        all_rules = self.get_all_rules()
        return [r for r in all_rules.design_rules if r.category == category]
    
    def _create_default_rules(self) -> RuleSet:
        """Create default rule set."""
        rules = [
            DesignRule(
                rule_id="R1_NAMING",
                name="Naming Convention",
                description="Tables and columns must use snake_case",
                category="naming",
                severity="warning"
            ),
            DesignRule(
                rule_id="R2_NO_NULL_FK",
                name="Non-nullable Foreign Keys",
                description="Foreign keys should be NOT NULL",
                category="structure",
                severity="warning"
            ),
            DesignRule(
                rule_id="R3_INDEX_FK",
                name="Index Foreign Keys",
                description="Foreign key columns should have indexes",
                category="performance",
                severity="info"
            ),
            DesignRule(
                rule_id="R4_PK_REQUIRED",
                name="Primary Key Required",
                description="All tables must have a primary key",
                category="structure",
                severity="error"
            ),
            DesignRule(
                rule_id="R5_TIMESTAMP_TRACKING",
                name="Timestamp Tracking",
                description="Tables should have created_at and updated_at columns",
                category="data_quality",
                severity="info"
            )
        ]
        
        return RuleSet(
            naming=NamingConvention(),
            design_rules=rules,
            type_mappings={
                DataType.STRING: "VARCHAR(255)",
                DataType.INTEGER: "INTEGER",
                DataType.DECIMAL: "DECIMAL(10,2)",
                DataType.BOOLEAN: "BOOLEAN",
                DataType.TIMESTAMP: "TIMESTAMP",
                DataType.UUID: "UUID"
            }
        )
    
    def _parse_rules(self, data: Dict[str, Any]) -> RuleSet:
        """Parse rules from JSON data."""
        # Implementation would parse JSON into RuleSet
        return self._default_rules
