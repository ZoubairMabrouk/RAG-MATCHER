from typing import List, Tuple, Dict
import numpy as np
from src.domain.repositeries.interfaces import IEmbeddingProvider
from src.domain.entities.schema import SchemaMetadata, Table, Column

class EmbeddingService:
    """
    Service for generating embeddings.
    Single Responsibility: Embedding generation only.
    """
    
    def __init__(self, provider: IEmbeddingProvider):
        self._provider = provider
    
    def embed_schema_objects(self, schema: SchemaMetadata) -> List[Tuple[str, np.ndarray, Dict]]:
        """Generate embeddings for all schema objects."""
        results = []
        
        for table in schema.tables:
            # Embed table
            text = self._table_to_text(table)
            embedding = self._provider.embed([text])[0]
            metadata = {
                "type": "table",
                "name": table.name,
                "schema": table.schema,
                "row_count": table.row_count
            }
            results.append((text, embedding, metadata))
            
            # Embed each column
            for col in table.columns:
                text = self._column_to_text(table.name, col)
                embedding = self._provider.embed([text])[0]
                metadata = {
                    "type": "column",
                    "table": table.name,
                    "name": col.name,
                    "data_type": col.data_type
                }
                results.append((text, embedding, metadata))
        
        return results
    
    def _table_to_text(self, table: Table) -> str:
        """Convert table to searchable text."""
        col_names = ", ".join([col.name for col in table.columns])
        return f"Table: {table.name}. Columns: {col_names}. Comment: {table.comment or 'None'}"
    
    def _column_to_text(self, table_name: str, col: Column) -> str:
        """Convert column to searchable text."""
        return f"Column: {table_name}.{col.name}. Type: {col.data_type}. Comment: {col.comment or 'None'}"

