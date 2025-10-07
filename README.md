# Database Evolution System

LLM-powered database schema evolution with RAG (Retrieval-Augmented Generation).

## Features

- 🎯 Analyzes U-Schema (NoSQL-oriented conceptual models)
- 🔍 Introspects current relational database schemas
- 🤖 Uses LLM (OpenAI/Anthropic) for intelligent migration planning
- 📚 RAG-powered context retrieval for accurate analysis
- ✅ Validates SQL and ensures migration safety
- 🏗️ Follows SOLID principles and Clean Architecture
- 🔒 Safe execution with dry-run mode and rollback support

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

## Usage

### CLI

```bash
# Analyze schema evolution
dbevolve analyze \
  --uschema schema.json \
  --connection "postgresql://user:pass@localhost/mydb" \
  --output evolution_plan.json

# Introspect current schema
dbevolve introspect \
  --connection "postgresql://user:pass@localhost/mydb" \
  --output current_schema.json

# Build RAG index
dbevolve index-schema \
  --connection "postgresql://user:pass@localhost/mydb"

# Start API server
dbevolve serve --port 8000
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

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_diff_engine.py
```

## License

MIT