from unittest.mock import Mock
from typing import List, Dict, Any
import numpy as np


from src.domain.entities.evolution import EvolutionPlan,SchemaChange
from src.domain.entities.schema import SchemaMetadata, Table
class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return fake embeddings."""
        return np.random.rand(len(texts), 1536)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, response: str = None):
        self.response = response or self._default_response()
    
    def generate_evolution_plan(self, context: Dict[str, Any]) -> 'EvolutionPlan':
        """Return mock evolution plan."""
        from src.domain.entities.evolution import EvolutionPlan
        from tests.fixtures.test_data import TestDataFactory
        
        return EvolutionPlan(
            changes=TestDataFactory.create_sample_changes(),
            description="Mock evolution plan",
            risk_level="low"
        )
    
    def generate_sql(self, change: 'SchemaChange') -> str:
        """Return mock SQL."""
        return f"-- Mock SQL for {change.change_type.value}"
    
    def _default_response(self) -> str:
        return '''
        {
          "description": "Mock evolution plan",
          "risk_level": "low",
          "changes": [],
          "backward_compatible": true
        }
        '''


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        self.data = []
    
    def add_embeddings(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """Store mock data."""
        for i, text in enumerate(texts):
            self.data.append({
                "text": text,
                "embedding": embeddings[i],
                "metadata": metadata[i]
            })
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """Return mock search results."""
        results = []
        for item in self.data[:top_k]:
            if filters:
                if all(item["metadata"].get(k) == v for k, v in filters.items()):
                    results.append(item)
            else:
                results.append(item)
        return results
    
    def delete_by_metadata(self, filters: Dict):
        """Mock delete."""
        self.data = [
            item for item in self.data
            if not all(item["metadata"].get(k) == v for k, v in filters.items())
        ]


class MockDatabaseInspector:
    """Mock database inspector for testing."""
    
    def __init__(self, schema: 'SchemaMetadata' = None):
        from tests.fixtures.test_data import TestDataFactory
        self.schema = schema or TestDataFactory.create_existing_schema()
    
    def introspect_schema(self) -> 'SchemaMetadata':
        """Return mock schema."""
        return self.schema
    
    def get_table_info(self, table_name: str) -> 'Table':
        """Return mock table info."""
        for table in self.schema.tables:
            if table.name == table_name:
                return table
        raise ValueError(f"Table {table_name} not found")

