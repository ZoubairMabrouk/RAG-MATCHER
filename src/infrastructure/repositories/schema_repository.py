from typing import Any, Dict
from src.infrastructure.database.inspector import IDataBaseInspector

from src.domain.repositeries.interfaces import ISchemaRepository

from src.domain.entities.schema import SchemaMetadata
class SchemaRepository(ISchemaRepository):
    """
    Repository for schema data access.
    Single Responsibility: Data access for schema.
    """
    
    def __init__(self, inspector: IDataBaseInspector):
        self._inspector = inspector
    
    def get_current_schema(self) -> SchemaMetadata:
        """Retrieve current database schema."""
        return self._inspector.introspect_schema()
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a specific table."""
        table = self._inspector.get_table_info(table_name)
        return {
            "row_count": table.row_count,
            "columns_count": len(table.columns),
            "indexes_count": len(table.indexes),
            "foreign_keys_count": len(table.foreign_keys)
        }
    
    def validate_sql(self, sql: str) -> bool:
        """Validate SQL syntax."""
        # Would use actual SQL parser
        return True
