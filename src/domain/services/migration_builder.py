from src.domain.entities.evolution import ChangeType, SchemaChange
from typing import List, Optional, Dict, Set
import logging

logger = logging.getLogger(__name__)
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
        
        # Remove duplicates and order changes by dependency
        deduplicated_changes = self._deduplicate_changes(changes)
        ordered_changes = self._order_changes(deduplicated_changes)
        
        logger.info(f"[MigrationBuilder] Processing {len(ordered_changes)} changes after deduplication")
        
        for change in ordered_changes:
            sql = self._generate_sql(change)
            if sql:
                sql_statements.append(sql)
                logger.debug(f"[MigrationBuilder] Generated SQL: {sql}")
        
        return sql_statements
    
    def _order_changes(self, changes: List[SchemaChange]) -> List[SchemaChange]:
        """Order changes to respect dependencies."""
        # Priority order
        priority = {
            ChangeType.CREATE_TABLE: 1,
            ChangeType.ADD_COLUMN: 2,
            ChangeType.MODIFY_COLUMN: 3,
            ChangeType.ADD_INDEX: 4,
            ChangeType.ADD_CONSTRAINT: 5
            # ChangeType.DROP_CONSTRAINT: 6,
            # ChangeType.DROP_INDEX: 7,
            # ChangeType.DROP_COLUMN: 8,
            # ChangeType.DROP_TABLE: 9
        }
        
        return sorted(changes, key=lambda c: priority.get(c.change_type, 99))
    
    def _deduplicate_changes(self, changes: List[SchemaChange]) -> List[SchemaChange]:
        """Remove duplicate changes to avoid SQL conflicts."""
        seen: Dict[str, SchemaChange] = {}
        
        for change in changes:
            # Create a unique key for the change
            key = f"{change.change_type.value}:{change.target_table}:{change.target_column or ''}"
            
            if key not in seen:
                seen[key] = change
            else:
                # Handle conflicts - prefer the more specific or later change
                existing = seen[key]
                if self._should_replace_change(existing, change):
                    seen[key] = change
                    logger.info(f"[MigrationBuilder] Replaced duplicate change: {key}")
                else:
                    logger.info(f"[MigrationBuilder] Skipped duplicate change: {key}")
        
        return list(seen.values())
    
    def _should_replace_change(self, existing: SchemaChange, new: SchemaChange) -> bool:
        """Determine if new change should replace existing one."""
        # Prefer MODIFY_COLUMN over ADD_COLUMN for the same column
        if (existing.change_type == ChangeType.ADD_COLUMN and 
            new.change_type == ChangeType.MODIFY_COLUMN and
            existing.target_table == new.target_table and
            existing.target_column == new.target_column):
            return True
        
        # Prefer changes with more specific definitions
        if (existing.change_type == new.change_type and
            existing.target_table == new.target_table and
            existing.target_column == new.target_column):
            return len(new.definition or "") > len(existing.definition or "")
        
        return False
    
    def _generate_sql(self, change: SchemaChange) -> Optional[str]:
        """Generate SQL for a single change."""
        if change.sql:
            return change.sql
        
        generators = {
            ChangeType.CREATE_TABLE: self._gen_create_table,
            #ChangeType.DROP_TABLE: self._gen_drop_table,
            ChangeType.ADD_COLUMN: self._gen_add_column,
            ChangeType.MODIFY_COLUMN: self._gen_modify_column,
            #ChangeType.DROP_COLUMN: self._gen_drop_column,
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
    
    # def _gen_drop_table(self, change: SchemaChange) -> str:
    #     """Generate DROP TABLE statement."""
    #     return f"DROP TABLE IF EXISTS {change.target_table};"
    
    def _gen_add_column(self, change: SchemaChange) -> str:
        """Generate ALTER TABLE ADD COLUMN statement."""
        defn = change.definition.strip()
        if defn.lower().startswith(change.target_column.lower()):
            # Definition already includes column name
            sql = f"ALTER TABLE {change.target_table} ADD COLUMN {defn};"
        else:
            # Definition is just the type, add column name
            sql = f"ALTER TABLE {change.target_table} ADD COLUMN {change.target_column} {defn};"
        return sql
    
    def _gen_modify_column(self, change: SchemaChange) -> str:
        """Generate ALTER TABLE MODIFY COLUMN statement."""
        if self._dialect == "postgresql":
            # PostgreSQL uses ALTER COLUMN ... TYPE
            return f"ALTER TABLE {change.target_table} ALTER COLUMN {change.target_column} {change.definition};"
        else:
            # Generic SQL (MySQL, etc.)
            return f"ALTER TABLE {change.target_table} MODIFY COLUMN {change.target_column} {change.definition};"
    
    # def _gen_drop_column(self, change: SchemaChange) -> str:
    #     """Generate ALTER TABLE DROP COLUMN statement."""
    #     return f"ALTER TABLE {change.target_table} DROP COLUMN {change.target_column};"
    
    def _gen_add_index(self, change: SchemaChange) -> str:
        """Generate CREATE INDEX statement."""
        idx_name = f"idx_{change.target_table}_{change.target_column}"
        return f"CREATE INDEX {idx_name} ON {change.target_table}({change.target_column});"
    
    def _gen_add_constraint(self, change: SchemaChange) -> str:
        """Generate ALTER TABLE ADD CONSTRAINT statement."""
        constraint_name = f"fk_{change.target_table}_{change.target_column}"
        return f"ALTER TABLE {change.target_table} ADD CONSTRAINT {constraint_name} {change.definition};"
