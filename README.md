# Database Evolution System with RAG Schema Matching

LLM-powered database schema evolution with advanced RAG (Retrieval-Augmented Generation) for NoSQL → SQL schema matching, specifically designed for MIMIC-III clinical data integration.

## Features

### Core Evolution System
- 🎯 Analyzes U-Schema (NoSQL-oriented conceptual models)
- 🔍 Introspects current relational database schemas
- 🤖 Uses LLM (OpenAI/Anthropic) for intelligent migration planning
- ✅ Validates SQL and ensures migration safety
- 🏗️ Follows SOLID principles and Clean Architecture
- 🔒 Safe execution with dry-run mode and rollback support

### Advanced RAG Schema Matching
- 📚 **Knowledge Base**: MIMIC-III schema with medical ontologies and synonyms
- 🔍 **Bi-Encoder + Cross-Encoder**: Fast retrieval with precise reranking
- 🧠 **LLM Orchestration**: Structured JSON output with validation
- ⚖️ **Hybrid Scoring**: Multi-signal scoring with calibration
- 🛡️ **Guardrails**: Type, unit, and constraint validation
- 📊 **Evaluation**: Comprehensive metrics and datasets
- 🔄 **Human-in-the-Loop**: Review workflow for uncertain matches
- 🔒 **Privacy-First**: Zero PHI, synthetic data only

## Architecture

```
┌─────────────────────────────────────────────┐
│         Presentation Layer                   │
│  (CLI, REST API, Web Dashboard)             │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         Application Layer                    │
│  (Use Cases, Orchestrators, DTOs)           │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         Domain Layer                         │
│  (Entities, Services, Repositories)         │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│       Infrastructure Layer                   │
│  (RAG, LLM, Database, Validators)           │
└─────────────────────────────────────────────┘
```

## Installation

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### 1. Setup RAG System

```bash
# Install dependencies
poetry install

# Setup RAG system with MIMIC-III knowledge base
python scripts/setup_rag_system.py

# This will:
# - Build MIMIC-III knowledge base from DDL and dictionary
# - Generate embeddings for all columns
# - Initialize vector store with FAISS
# - Create demo dataset for testing
```

### 2. RAG Schema Matching

```bash
# Start API server with RAG endpoints
uvicorn src.presentation.api.app:app --reload

# Test single field matching
curl -X POST http://localhost:8000/api/v1/rag/match/single \
  -H "Content-Type: application/json" \
  -d '{
    "path": "patient.heart_rate",
    "name_tokens": ["heart", "rate"],
    "inferred_type": "integer",
    "units": "bpm",
    "hints": ["vital signs", "cardiac"]
  }'

# Batch field matching
curl -X POST http://localhost:8000/api/v1/rag/match/batch \
  -H "Content-Type: application/json" \
  -d '{
    "fields": [
      {"path": "patient.id", "name_tokens": ["patient", "id"], "inferred_type": "id"},
      {"path": "admission.date", "name_tokens": ["admission", "date"], "inferred_type": "datetime"}
    ]
  }'
```

### 3. Traditional Schema Evolution

```bash
# Analyze schema evolution (existing functionality)
dbevolve analyze \
  --uschema schema.json \
  --connection "postgresql://user:pass@localhost/mydb" \
  --output evolution_plan.json

# Introspect current schema
dbevolve introspect \
  --connection "postgresql://user:pass@localhost/mydb" \
  --output current_schema.json
```

### API

```bash
# Start server
uvicorn src.presentation.api.app:app --reload

# Example request
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "uschema": {...},
    "connection_string": "postgresql://...",
    "use_rag": true,
    "dry_run": true
  }'
```

### Python

```python
from src.application.dtos.evolution_dto import EvolutionRequest
from src.infrastructure.di_container import DIContainer

# Create request
request = EvolutionRequest(
    uschema_json=uschema_data,
    database_connection="postgresql://...",
    include_rag=True,
    dry_run=True
)

# Process evolution
container = DIContainer()
container.configure(request.database_connection, "postgresql")
orchestrator = container.get_orchestrator()

response = orchestrator.process_evolution(request)
print(f"Generated {len(response.sql_statements)} SQL statements")
```

## Design Principles

### SOLID Principles

1. **Single Responsibility**: Each class has one reason to change
   - `DiffEngine`: Only computes schema differences
   - `RuleEngine`: Only validates against rules
   - `MigrationBuilder`: Only generates SQL

2. **Open/Closed**: Open for extension, closed for modification
   - New database dialects via strategy pattern
   - New LLM providers via interface implementation

3. **Liskov Substitution**: Interfaces are properly abstracted
   - `ILLMClient`, `IVectorStore`, `IDataBaseInspector`

4. **Interface Segregation**: Small, focused interfaces
   - Separate repository interfaces for each concern

5. **Dependency Inversion**: Depend on abstractions
   - Services depend on repository interfaces
   - Infrastructure provides implementations

## RAG Schema Matching Documentation

For detailed documentation on the RAG schema matching system, see:
- **[RAG System Guide](docs/RAG_SYSTEM.md)** - Complete implementation guide
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI
- **[Examples](examples/rag_matching_example.py)** - Usage examples

### Key RAG Features

1. **MIMIC-III Knowledge Base**: Pre-built corpus with medical ontologies
2. **Hybrid Scoring**: Bi-encoder + Cross-encoder + LLM confidence
3. **Guardrails**: Type, unit, and constraint validation
4. **Human Review**: Workflow for uncertain matches
5. **Privacy-First**: Zero PHI, synthetic data only
6. **Evaluation**: Comprehensive metrics and test datasets

### Decision Actions

- **ACCEPT**: High-confidence match, ready for production
- **REVIEW**: Requires human validation
- **REJECT**: Fall back to rule-based engine

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run RAG-specific tests
pytest tests/integration/test_rag_integration.py

# Run specific test
pytest tests/test_diff_engine.py
```

## Architecture Components

### RAG System
- `src/domain/entities/rag_schema.py` - Core entities and types
- `src/infrastructure/rag/knowledge_base_builder.py` - Knowledge base construction
- `src/infrastructure/rag/embedding_service.py` - Bi-encoder and cross-encoder
- `src/infrastructure/rag/vector_store.py` - FAISS vector storage
- `src/infrastructure/rag/retriever.py` - Advanced retrieval with filtering
- `src/infrastructure/rag/llm_orchestrator.py` - LLM coordination and validation
- `src/infrastructure/rag/scoring_system.py` - Hybrid scoring and calibration
- `src/infrastructure/rag/rag_orchestrator.py` - Main orchestrator
- `src/infrastructure/rag/evaluation_metrics.py` - Evaluation and metrics
- `src/presentation/api/rag_endpoints.py` - REST API endpoints

### Traditional System
- `src/application/` - Use cases and orchestrators
- `src/domain/` - Entities and business logic
- `src/infrastructure/` - External services and repositories
- `src/presentation/` - CLI and API interfaces

## License

MIT