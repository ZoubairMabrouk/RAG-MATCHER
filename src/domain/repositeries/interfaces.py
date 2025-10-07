from abc import ABC, abstractmethod
from typing import List ,Dict,Any, Optional
from src.domain.entities.schema import SchemaMetadata, USchema
from src.domain.entities.rules import DesignRule, RuleSet
from src.domain.entities.evolution import SchemaChange    
from src.application.dtos.evolution_dto import EvolutionPlan

class ISchemaRepository(ABC):
    """Interface for schema data access."""
    
    @abstractmethod
    def get_current_schema(self) -> SchemaMetadata:
        """Retrieve current database schema."""
        pass
    
    @abstractmethod
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a specific table."""
        pass
    
    @abstractmethod
    def validate_sql(self, sql: str) -> bool:
        """Validate SQL syntax."""
        pass

class IRuleRepository(ABC):
    """Interface for rule data access."""
    
    @abstractmethod
    def get_all_rules(self) -> RuleSet:
        """Retrieve all design rules."""
        pass
    
    @abstractmethod
    def get_rules_by_category(self, category: str) -> List[DesignRule]:
        """Retrieve rules by category."""
        pass
"""Embedding service for RAG."""

import numpy as np
from typing import List


class IEmbeddingProvider(ABC):
    """Interface for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass


class OpenAIEmbeddingProvider(IEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._api_key = api_key
        self._model = model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        # Implementation would use openai library
        import openai
        response = openai.Embedding.create(input=texts, model=self._model)
        return np.array([item['embedding'] for item in response['data']])
        
        # Placeholder
        # return np.random.rand(len(texts), 1536)
class ILLMClient(ABC):
    """Interface for LLM interactions."""
    
    @abstractmethod
    def generate_evolution_plan(self, context: Dict[str, Any]) -> EvolutionPlan:
        """Generate evolution plan using LLM."""
        pass

    @abstractmethod
    def generate_sql(self, change: SchemaChange) -> str:
        """Generate SQL for a schema change."""
        pass

class IUSchemaRepository(ABC):
    """Interface for U-Schema access."""
    
    @abstractmethod
    def parse_uschema(self, json_data: Dict[str, Any]) -> USchema:
        """Parse U-Schema from JSON."""
        pass
    
    @abstractmethod
    def validate_uschema(self, schema: USchema) -> bool:
        """Validate U-Schema structure."""
        pass
class IVectorStore(ABC):
    """Interface for vector store operations."""
    
    @abstractmethod
    def add_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar items."""
        pass
    
    @abstractmethod
    def delete_by_metadata(self, filters: Dict[str, Any]) -> None:
        """Delete items by metadata filters."""
        pass
