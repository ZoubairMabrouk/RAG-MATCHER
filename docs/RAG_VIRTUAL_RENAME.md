# RAG-based Virtual Renaming System

## Overview

This document describes the new RAG (Retrieval-Augmented Generation) based virtual renaming system that replaces heuristic-based detection (Jaccard similarity, name matching) with semantic matching using embeddings and optional LLM validation.

## Key Features

- **Semantic Matching**: Uses embeddings to understand semantic relationships between U-Schema entities and existing database tables/columns
- **Virtual Renaming**: Maps entities to existing tables without generating physical `RENAME` operations
- **LLM Validation**: Optional OpenAI LLM integration for enhanced confidence scoring and rationale
- **No Physical Renames**: System never emits `ALTER TABLE ... RENAME TO ...` statements
- **Fallback Support**: Falls back to heuristic matching if RAG matcher is unavailable

## Architecture

### Core Components

1. **RAGSchemaMatcher** (`src/infrastructure/rag/rag_schema_matcher.py`)
   - Main component for semantic matching
   - Builds knowledge base from current schema
   - Performs vector similarity search
   - Optional LLM validation

2. **MatchResult** (dataclass)
   - Contains matching results with confidence scores
   - Includes rationale and metadata

3. **Enhanced DiffEngine** (`src/domain/services/diff_engine.py`)
   - Integrated with RAG matcher
   - Uses virtual aliasing instead of heuristic detection
   - Maintains backward compatibility

4. **Enhanced MigrationBuilder** (`src/domain/services/migration_builder.py`)
   - Handles `MODIFY_COLUMN` operations
   - Deduplication logic to avoid SQL conflicts
   - PostgreSQL-specific syntax support

## Configuration

### Environment Variables

```bash
# Enable LLM validation (optional)
RAG_USE_LLM=1

# OpenAI API key (required if RAG_USE_LLM=1)
OPENAI_API_KEY=your_api_key_here

# Embedding provider (local or openai)
RAG_EMBED_PROVIDER=local  # or openai

# Local embedding model
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# OpenAI embedding model
EMBEDDING_MODEL=text-embedding-3-small
```

### Thresholds

- **Table matching threshold**: 0.62 (default)
- **Column matching threshold**: 0.68 (default)

These can be adjusted in the `RAGSchemaMatcher` constructor.

## Usage Examples

### Basic Usage

```python
from src.infrastructure.di_container import DIContainer

# Configure container
container = DIContainer()
container.configure("postgresql://user:pass@localhost:5432/db")

# Get diff engine with RAG matcher
diff_engine = container.get_diff_engine()

# Compute differences with virtual renaming
changes = diff_engine.compute_diff(uschema, current_schema)
```

### Direct RAG Matcher Usage

```python
# Get RAG matcher
matcher = container.get_rag_schema_matcher()

# Match table
table_result = matcher.match_table("items", ["id", "name", "price"])
print(f"Matched to: {table_result.target_name} (confidence: {table_result.confidence})")

# Match column
column_result = matcher.match_column("products", "qte", "INTEGER")
print(f"Matched to: {column_result.target_name} (confidence: {column_result.confidence})")
```

## Example Scenario

### Input: U-Schema Entity
```json
{
  "name": "items",
  "attributes": [
    {"name": "id", "type": "INTEGER"},
    {"name": "name", "type": "STRING"},
    {"name": "price", "type": "DECIMAL"},
    {"name": "qte", "type": "INTEGER"},
    {"name": "ref", "type": "STRING"}
  ]
}
```

### Current Schema
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    quantity INTEGER,
    reference VARCHAR(255)
);
```

### Expected Output

**Before (Heuristic)**:
```sql
CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    qte INTEGER,
    ref VARCHAR(255)
);
```

**After (RAG Virtual Rename)**:
```sql
-- No CREATE TABLE items
-- Virtual mapping: items -> products
-- Column mappings: qte -> quantity, ref -> reference
```

**Generated Migration**:
```sql
-- No physical rename operations
-- Only modifications to existing products table
ALTER TABLE products ALTER COLUMN quantity TYPE INTEGER;  -- if type differs
ALTER TABLE products ALTER COLUMN reference TYPE VARCHAR(255);  -- if type differs
```

## Testing

### Run Demo Script

```bash
# Basic demo (retrieval-only)
python examples/run_rag_virtual_rename_demo.py

# With LLM validation
RAG_USE_LLM=1 OPENAI_API_KEY=your_key python examples/run_rag_virtual_rename_demo.py
```

### Run Tests

```bash
# Run specific test
python examples/test_rag_virtual_rename.py

# Run with different configurations
RAG_USE_LLM=0 python examples/test_rag_virtual_rename.py
```

## Performance Considerations

### Embedding Models

- **Local (all-MiniLM-L6-v2)**: 384 dimensions, fast, no API costs
- **OpenAI (text-embedding-3-small)**: 1536 dimensions, high quality, API costs

### Vector Store

- Uses FAISS with different index types:
  - `Flat`: Exact search, slower for large datasets
  - `IVF_PQ`: Approximate search, faster for large datasets
  - `HNSW`: Hierarchical search, good balance

### LLM Validation

- Optional but recommended for production
- Adds API latency and costs
- Provides better confidence scoring and rationale

## Troubleshooting

### Common Issues

1. **No matches found**
   - Check threshold settings
   - Verify knowledge base was built correctly
   - Ensure embeddings are properly generated

2. **Low confidence scores**
   - Adjust thresholds
   - Enable LLM validation
   - Check embedding quality

3. **Knowledge base build failures**
   - Verify database connection
   - Check schema inspection permissions
   - Ensure proper table/column metadata

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check matcher statistics:
```python
matcher = container.get_rag_schema_matcher()
stats = matcher.get_statistics()
print(stats)
```

## Migration from Heuristic System

The system is backward compatible. To migrate:

1. **Enable RAG matcher** in DI container (automatic)
2. **Set environment variables** as needed
3. **Test with existing schemas** to verify behavior
4. **Adjust thresholds** if needed for your domain

The system will automatically fall back to heuristic matching if RAG matcher is unavailable.

## Future Enhancements

- **Custom embedding models** for domain-specific schemas
- **Multi-language support** for international schemas
- **Confidence learning** from user feedback
- **Batch processing** for large schema migrations
- **Integration with schema evolution tools**

## API Reference

### RAGSchemaMatcher

```python
class RAGSchemaMatcher:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: RAGVectorStore,
        llm_client: Optional[OpenAILLMClient] = None,
        table_accept_threshold: float = 0.62,
        column_accept_threshold: float = 0.68,
        top_k_search: int = 5
    )
    
    def build_kb(self, schema: SchemaMetadata) -> List[KnowledgeBaseDocument]
    def index_kb(self, documents: List[KnowledgeBaseDocument]) -> None
    def match_table(self, entity_name: str, attributes: List[str]) -> MatchResult
    def match_column(self, table_name: str, attr_name: str, attr_type: str) -> MatchResult
    def get_statistics(self) -> Dict[str, Any]
```

### MatchResult

```python
@dataclass
class MatchResult:
    target_name: Optional[str]
    confidence: float
    rationale: str
    extra: Dict[str, Any]
```

## Contributing

When contributing to the RAG virtual rename system:

1. **Maintain backward compatibility** with heuristic fallback
2. **Add comprehensive tests** for new features
3. **Update documentation** for API changes
4. **Consider performance implications** of embedding operations
5. **Test with both local and OpenAI embeddings**
