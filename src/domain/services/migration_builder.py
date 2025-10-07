from src.domain.entities.evolution import ChangeType, SchemaChange
from typing import List, Optional
class MigrationBuilder:
    """
    Generates SQL DDL from schema changes.
    Single Responsibility: SQL generation only.
    """
    
    def __init__(self, dialect: str = "postgresql"):
        self._dialect = dialect
    
    def build_migration(self, changes: List[SchemaChange]) -> List[str]:
        """Generate SQL statements from changes."""
        sql_statements = []
        
        # Order changes by dependency
        ordered_changes = self._order_changes(changes)
        
        for change in ordered_changes:
            sql = self._generate_sql(change)
            if sql:
                sql_statements.append(sql)
        
        return sql_statements
    
    def _order_changes(self, changes: List[SchemaChange]) -> List[SchemaChange]:
        """Order changes to respect dependencies."""
        # Priority order
        priority = {
            ChangeType.CREATE_TABLE: 1,
            ChangeType.ADD_COLUMN: 2,
            ChangeType.ADD_INDEX: 3,
            ChangeType.ADD_CONSTRAINT: 4,
            ChangeType.MODIFY_COLUMN: 5,
            ChangeType.DROP_CONSTRAINT: 6,
            ChangeType.DROP_INDEX: 7,
            ChangeType.DROP_COLUMN: 8,
            ChangeType.DROP_TABLE: 9
        }
        
        return sorted(changes, key=lambda c: priority.get(c.change_type, 99))
    
    def _generate_sql(self, change: SchemaChange) -> Optional[str]:
        """Generate SQL for a single change."""
        if change.sql:
            return change.sql
        
        generators = {
            ChangeType.CREATE_TABLE: self._gen_create_table,
            ChangeType.DROP_TABLE: self._gen_drop_table,
            ChangeType.ADD_COLUMN: self._gen_add_column,
            ChangeType.DROP_COLUMN: self._gen_drop_column,
            ChangeType.ADD_INDEX: self._gen_add_index,
            ChangeType.ADD_CONSTRAINT: self._gen_add_constraint,
        }
        
        generator = generators.get(change.change_type)
        if generator:
            return generator(change)
        
        return None
    
    def _gen_create_table(self, change: SchemaChange) -> str:
        """Generate CREATE TABLE statement."""
        return f"CREATE TABLE {change.target_table} ({change.definition});"
    
    def _gen_drop_table(self, change: SchemaChange) -> str:
        """Generate DROP TABLE statement."""
        return f"DROP TABLE IF EXISTS {change.target_table};"
    
    def _gen_add_column(self, change: SchemaChange) -> str:
        """Generate ALTER TABLE ADD COLUMN statement."""
        return f"ALTER TABLE {change.target_table} ADD COLUMN {change.target_column} {change.definition};"
    
    def _gen_drop_column(self, change: SchemaChange) -> str:
        """Generate ALTER TABLE DROP COLUMN statement."""
        return f"ALTER TABLE {change.target_table} DROP COLUMN {change.target_column};"
    
    def _gen_add_index(self, change: SchemaChange) -> str:
        """Generate CREATE INDEX statement."""
        idx_name = f"idx_{change.target_table}_{change.target_column}"
        return f"CREATE INDEX {idx_name} ON {change.target_table}({change.target_column});"
    
    def _gen_add_constraint(self, change: SchemaChange) -> str:
        """Generate ALTER TABLE ADD CONSTRAINT statement."""
        constraint_name = f"fk_{change.target_table}_{change.target_column}"
        return f"ALTER TABLE {change.target_table} ADD CONSTRAINT {constraint_name} {change.definition};"
