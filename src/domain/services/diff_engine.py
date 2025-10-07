from typing import List, Tuple

from src.domain.entities.evolution import ChangeType, SchemaChange
from src.domain.entities.schema import USchema, SchemaMetadata, USchemaEntity, USchemaAttribute, Table
from src.domain.entities.rules import NamingConvention,DataType
class DiffEngine:
    """
    Computes symbolic differences between U-Schema and current RDB schema.
    Single Responsibility: Only handles diff computation.
    """
    
    def __init__(self, naming_convention: NamingConvention):
        self._naming = naming_convention
    
    def compute_diff(self, uschema: USchema, current_schema: SchemaMetadata) -> List[SchemaChange]:
        """
        Compute differences between U-Schema and current database schema.
        Returns a list of required changes.
        """
        changes = []
        
        # Build lookup maps for efficient comparison
        existing_tables = {table.name for table in current_schema.tables}
        table_map = {table.name: table for table in current_schema.tables}
        
        for entity in uschema.entities:
            table_name = self._entity_to_table_name(entity.name)
            
            if table_name not in existing_tables:
                # Need to create new table
                changes.extend(self._create_table_changes(entity, table_name))
            else:
                # Check for column differences
                changes.extend(self._compare_columns(entity, table_map[table_name]))
                
                # Check for relationship differences (foreign keys)
                changes.extend(self._compare_relationships(entity, table_map[table_name]))
        
        # Check for tables that exist but not in U-Schema (potential drops)
        uschema_tables = {self._entity_to_table_name(e.name) for e in uschema.entities}
        for table_name in existing_tables:
            if table_name not in uschema_tables:
                changes.append(SchemaChange(
                    change_type=ChangeType.DROP_TABLE,
                    target_table=table_name,
                    reason=f"Table '{table_name}' not present in U-Schema",
                    safe=False,
                    estimated_impact="high"
                ))
        
        return changes
    
    def _entity_to_table_name(self, entity_name: str) -> str:
        """Convert entity name to table name using naming convention."""
        # Convert PascalCase to snake_case and pluralize
        import re
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', entity_name).lower()
        # Simple pluralization (production would use inflect library)
        if not snake.endswith('s'):
            snake += 's'
        return snake
    
    def _create_table_changes(self, entity: USchemaEntity, table_name: str) -> List[SchemaChange]:
        """Generate changes to create a new table."""
        changes = []
        
        # Create table
        column_defs = []
        for attr in entity.attributes:
            col_def = self._attribute_to_column_def(attr)
            column_defs.append(col_def)
        
        changes.append(SchemaChange(
            change_type=ChangeType.CREATE_TABLE,
            target_table=table_name,
            definition=", ".join(column_defs),
            reason=f"Entity '{entity.name}' requires new table",
            safe=True,
            estimated_impact="medium"
        ))
        
        return changes
    
    def _attribute_to_column_def(self, attr: USchemaAttribute) -> str:
        """Convert U-Schema attribute to SQL column definition."""
        type_map = {
            DataType.STRING: "VARCHAR(255)",
            DataType.INTEGER: "INTEGER",
            DataType.DECIMAL: "DECIMAL(10,2)",
            DataType.BOOLEAN: "BOOLEAN",
            DataType.TIMESTAMP: "TIMESTAMP",
            DataType.DATE: "DATE",
            DataType.JSON: "JSONB",
            DataType.UUID: "UUID"
        }
        
        sql_type = type_map.get(attr.data_type, "TEXT")
        null_clause = "NOT NULL" if attr.required else ""
        
        return f"{attr.name} {sql_type} {null_clause}".strip()
    
    def _compare_columns(self, entity: USchemaEntity, table: Table) -> List[SchemaChange]:
        """Compare entity attributes with table columns."""
        changes = []
        
        existing_columns = {col.name for col in table.columns}
        required_columns = {attr.name for attr in entity.attributes}
        
        # Columns to add
        for attr in entity.attributes:
            if attr.name not in existing_columns:
                changes.append(SchemaChange(
                    change_type=ChangeType.ADD_COLUMN,
                    target_table=table.name,
                    target_column=attr.name,
                    definition=self._attribute_to_column_def(attr),
                    reason=f"Attribute '{attr.name}' not present in table '{table.name}'",
                    safe=True,
                    estimated_impact="low"
                ))
        
        # Columns to drop
        for col in table.columns:
            if col.name not in required_columns:
                changes.append(SchemaChange(
                    change_type=ChangeType.DROP_COLUMN,
                    target_table=table.name,
                    target_column=col.name,
                    reason=f"Column '{col.name}' not in U-Schema",
                    safe=False,
                    requires_data_migration=True,
                    estimated_impact="high"
                ))
        
        return changes
    
    def _compare_relationships(self, entity: USchemaEntity, table: Table) -> List[SchemaChange]:
        """Compare entity relationships with foreign keys."""
        changes = []
        
        existing_fks = {fk.column: fk for fk in table.foreign_keys}
        
        for rel in entity.relationships:
            if rel.relationship_type == "belongsTo" and rel.foreign_key:
                if rel.foreign_key not in existing_fks:
                    ref_table = self._entity_to_table_name(rel.target_entity)
                    changes.append(SchemaChange(
                        change_type=ChangeType.ADD_CONSTRAINT,
                        target_table=table.name,
                        target_column=rel.foreign_key,
                        definition=f"FOREIGN KEY ({rel.foreign_key}) REFERENCES {ref_table}(id)",
                        reason=f"Relationship to '{rel.target_entity}' requires foreign key",
                        safe=True,
                        estimated_impact="low"
                    ))
        
        return changes
