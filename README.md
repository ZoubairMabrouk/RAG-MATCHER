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

### RAG-based Virtual Renaming (NEW!)
- 🎯 **Semantic Matching**: Uses embeddings to understand relationships between U-Schema entities and existing tables
- 🔄 **Virtual Renaming**: Maps entities to existing tables without physical `RENAME` operations
- 🧠 **LLM Validation**: Optional OpenAI integration for enhanced confidence scoring
- 📊 **Confidence Scoring**: Configurable thresholds for table (0.62) and column (0.68) matching
- 🔄 **Fallback Support**: Falls back to heuristic matching if RAG unavailable
- 🚫 **No Physical Renames**: Never emits `ALTER TABLE ... RENAME TO ...` statements

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

### Prerequisites

- Docker Desktop with Docker Compose v2 (`docker compose`)
- At least 8 GB of available RAM for the Ollama and embedding services

The project is designed to run in Docker. Poetry and a local Python installation are not required.

### Configuration

Create a `.env` file if you need to provide optional API keys:

```bash
# Copy the example file when it exists
cp .env.example .env

# Edit .env and add your OpenAI or Anthropic API key when required
```

The default Docker Compose configuration starts:

- PostgreSQL at `localhost:55432` (`test` / `test`, database `test`)
- Ollama at `http://localhost:11434`
- The application container with the project mounted at `/app`

## Quick Start

### 1. Build and start the services

```bash
# Build the application image and start PostgreSQL and Ollama
docker compose up -d --build db ollama

# Check the service status
docker compose ps
```

### 2. Setup RAG System

```bash
# Run the RAG setup inside the application container
docker compose run --rm app python scripts/setup_rag_system.py

# Setup the optional LLM-enabled environment
docker compose run --rm app python scripts/setup_rag_environment.py --llm --api-key YOUR_KEY

# This will:
# - Build MIMIC-III knowledge base from DDL and dictionary
# - Generate embeddings for all columns
# - Initialize vector store with FAISS
# - Create demo dataset for testing
```

### 3. RAG Schema Matching API

```bash
# Start the API in a temporary application container
docker compose run --rm --service-ports -p 8000:8000 app \
  uvicorn src.presentation.api.app:app --host 0.0.0.0 --port 8000

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

The interactive API documentation is available at <http://localhost:8000/docs>.

### 4. RAG Virtual Renaming (NEW!)

```bash
# Validate implementation
docker compose run --rm app python scripts/validate_rag_implementation.py

# Run demo with virtual renaming
docker compose run --rm app python examples/run_rag_virtual_rename_demo.py

# Test specific scenarios
docker compose run --rm app python examples/test_rag_virtual_rename.py
```

### 5. Traditional Schema Evolution

```bash
# Analyze schema evolution inside the application container
docker compose run --rm app dbevolve analyze \
  --uschema schema.json \
  --connection "postgresql://test:test@db:5432/test" \
  --output evolution_plan.json

# Introspect current schema
docker compose run --rm app dbevolve introspect \
  --connection "postgresql://test:test@db:5432/test" \
  --output current_schema.json
```

### Stop the services

```bash
# Stop containers while preserving PostgreSQL and Ollama data
docker compose down

# Stop containers and delete their data volumes
docker compose down -v
```

### API

```bash
# Start the API as described above
docker compose run --rm --service-ports -p 8000:8000 app \
  uvicorn src.presentation.api.app:app --host 0.0.0.0 --port 8000

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