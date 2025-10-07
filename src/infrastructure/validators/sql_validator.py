"""SQL validation services."""

import sqlparse
from typing import Tuple, Optional, List
from src.domain.entities.evolution import SchemaChange, ChangeType
from src.infrastructure.database.inspector import IDataBaseInspector


class SQLValidator:
    """
    Validates SQL statements.
    Single Responsibility: SQL validation.
    """
    
    def __init__(self, dialect: str = "postgresql"):
        self._dialect = dialect
    
    def validate_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax.
        Returns (is_valid, error_message).
        """
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql)
            
            if not parsed:
                return False, "Empty SQL statement"
            
            # Check for common issues
            for statement in parsed:
                # Check for dangerous operations
                tokens = [t.value.upper() for t in statement.flatten()]
                
                if 'DROP' in tokens and 'DATABASE' in tokens:
                    return False, "DROP DATABASE is not allowed"
                
                if 'TRUNCATE' in tokens:
                    return False, "TRUNCATE requires explicit approval"
            
            return True, None
            
        except Exception as e:
            return False, f"Syntax error: {str(e)}"
    
    def validate_safety(self, change: SchemaChange) -> Tuple[bool, List[str]]:
        """
        Check if a change is safe to execute.
        Returns (is_safe, list_of_warnings).
        """
        warnings = []
        
        # Check for destructive operations
        if change.change_type in [ChangeType.DROP_TABLE, ChangeType.DROP_COLUMN]:
            warnings.append(f"Destructive operation: {change.change_type.value}")
        
        # Check for NOT NULL on existing column
        if change.change_type == ChangeType.ADD_COLUMN:
            if change.definition and 'NOT NULL' in change.definition and not 'DEFAULT' in change.definition:
                warnings.append("Adding NOT NULL column without DEFAULT may fail on existing data")
        
        # Check for large table modifications
        if change.estimated_impact == "high":
            warnings.append("High impact operation - may take long time and lock table")
        
        is_safe = len(warnings) == 0 or change.safe
        return is_safe, warnings


class SafetyValidator:
    """
    Validates safety of migrations.
    Single Responsibility: Safety validation.
    """
    
    def __init__(self, inspector: IDataBaseInspector):
        self._inspector = inspector
    
    def check_data_compatibility(self, change: SchemaChange) -> Tuple[bool, Optional[str]]:
        """Check if change is compatible with existing data."""
        if change.change_type == ChangeType.MODIFY_COLUMN:
            # Check if type conversion is safe
            table = self._inspector.get_table_info(change.target_table)
            
            for col in table.columns:
                if col.name == change.target_column:
                    # Check if conversion is safe
                    if not self._is_safe_type_conversion(col.data_type, change.definition):
                        return False, f"Unsafe type conversion from {col.data_type} to {change.definition}"
            
        return True, None
    
    def _is_safe_type_conversion(self, from_type: str, to_type: str) -> bool:
        """Check if type conversion is safe."""
        # Simplified logic - production would be more sophisticated
        safe_conversions = {
            'integer': ['bigint', 'numeric', 'text'],
            'varchar': ['text'],
            'date': ['timestamp']
        }
        
        from_base = from_type.split('(')[0].lower()
        to_base = to_type.split('(')[0].lower()
        
        if from_base == to_base:
            return True
        
        return to_base in safe_conversions.get(from_base, [])
    
    def estimate_migration_duration(self, changes: List[SchemaChange]) -> int:
        """Estimate migration duration in minutes."""
        total_minutes = 0
        
        for change in changes:
            if change.change_type == ChangeType.CREATE_TABLE:
                total_minutes += 1
            elif change.change_type == ChangeType.ADD_COLUMN:
                # Get table size
                table = self._inspector.get_table_info(change.target_table)
                if table.row_count > 1000000:
                    total_minutes += 5
                else:
                    total_minutes += 1
            elif change.change_type == ChangeType.ADD_INDEX:
                table = self._inspector.get_table_info(change.target_table)
                if table.row_count > 1000000:
                    total_minutes += 10
                else:
                    total_minutes += 2
        
        return max(1, total_minutes)