# ==============================================================================
# DB EVOLUTION SYSTEM - PART 3
# Presentation Layer, CLI, API, Tests & Project Setup
# ==============================================================================

# ==============================================================================
# FILE: src/presentation/cli/commands.py
# ==============================================================================
"""CLI commands using Click framework."""

import click
import json
from pathlib import Path
from typing import Optional
import numpy as np

from src.application.orchestrators.evolution_orchestrator import EvolutionOrchestrator
from src.application.dtos.evolution_dto import EvolutionRequest
from src.infrastructure.di_container import DIContainer


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Database Evolution System - LLM-powered schema migration tool."""
    pass


@cli.command()
@click.option('--uschema', '-u', required=True, type=click.Path(exists=True),
              help='Path to U-Schema JSON file')
@click.option('--connection', '-c', required=True,
              help='Database connection string')
@click.option('--output', '-o', type=click.Path(), default='evolution_plan.json',
              help='Output file for evolution plan')
@click.option('--dry-run/--execute', default=True,
              help='Dry run (default) or execute migration')
@click.option('--use-rag/--no-rag', default=True,
              help='Use RAG for context retrieval')
@click.option('--dialect', type=click.Choice(['postgresql', 'mysql', 'sqlite']),
              default='postgresql', help='Target database dialect')
def analyze(uschema, connection, output, dry_run, use_rag, dialect):
    """Analyze schema and generate evolution plan."""
    
    click.echo("🚀 Database Evolution System")
    click.echo("=" * 50)
    
    # Load U-Schema
    with open(uschema, 'r') as f:
        uschema_data = json.load(f)
    
    # Create request
    request = EvolutionRequest(
        uschema_json=uschema_data,
        database_connection=connection,
        include_rag=use_rag,
        dry_run=dry_run,
        safe_mode=True,
        target_dialect=dialect
    )
    
    # Get orchestrator from DI container
    container = DIContainer()
    container.configure(connection, dialect)
    orchestrator = container.get_orchestrator()
    
    # Process evolution
    try:
        response = orchestrator.process_evolution(request)
        
        # Display results
        click.echo(f"\n📊 Evolution Plan Summary:")
        click.echo(f"   Changes: {len(response.plan.changes)}")
        click.echo(f"   Risk Level: {response.plan.risk_level}")
        click.echo(f"   Estimated Duration: {response.estimated_duration_minutes} minutes")
        click.echo(f"   Backward Compatible: {response.plan.backward_compatible}")
        
        # Show changes
        click.echo(f"\n📝 Proposed Changes:")
        for i, change in enumerate(response.plan.changes, 1):
            icon = "✓" if change.safe else "⚠️"
            click.echo(f"   {icon} {i}. {change.change_type.value}: {change.target_table}")
            click.echo(f"      Reason: {change.reason}")
        
        # Show SQL
        click.echo(f"\n💾 Generated SQL:")
        for sql in response.sql_statements:
            click.echo(f"   {sql}")
        
        # Validation results
        if response.validation_results:
            click.echo(f"\n✅ Validation:")
            for key, value in response.validation_results.items():
                click.echo(f"   {key}: {value}")
        
        # Save to file
        output_data = {
            "description": response.plan.description,
            "risk_level": response.plan.risk_level,
            "changes": [
                {
                    "type": c.change_type.value,
                    "table": c.target_table,
                    "column": c.target_column,
                    "reason": c.reason,
                    "sql": c.sql
                }
                for c in response.plan.changes
            ],
            "sql_statements": response.sql_statements,
            "validation": response.validation_results
        }
        
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        click.echo(f"\n💾 Plan saved to: {output}")
        
        if response.execution_report:
            click.echo(f"\n📄 Execution Report:")
            click.echo(response.execution_report)
        
        click.echo(f"\n{'✅ Analysis complete!' if dry_run else '✅ Migration complete!'}")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--connection', '-c', required=True,
              help='Database connection string')
@click.option('--output', '-o', type=click.Path(), default='schema.json',
              help='Output file for schema dump')
def introspect(connection, output):
    """Introspect current database schema."""
    
    click.echo("🔍 Introspecting database...")
    
    from src.infrastructure.database.inspector import PostgresInspector
    
    inspector = PostgresInspector(connection)
    schema = inspector.introspect_schema()
    
    # Convert to dict
    schema_dict = {
        "database": schema.database_name,
        "version": schema.version,
        "tables": [
            {
                "name": t.name,
                "schema": t.schema,
                "columns": [
                    {
                        "name": c.name,
                        "type": c.data_type,
                        "nullable": c.nullable,
                        "pk": c.primary_key
                    }
                    for c in t.columns
                ],
                "foreign_keys": [
                    {
                        "column": fk.column,
                        "references": f"{fk.referenced_table}({fk.referenced_column})"
                    }
                    for fk in t.foreign_keys
                ],
                "row_count": t.row_count
            }
            for t in schema.tables
        ]
    }
    
    with open(output, 'w') as f:
        json.dump(schema_dict, f, indent=2)
    
    click.echo(f"✅ Schema saved to: {output}")
    click.echo(f"   Tables: {len(schema.tables)}")


@cli.command()
@click.option('--connection', '-c', required=True,
              help='Database connection string')
@click.option('--rules', '-r', type=click.Path(exists=True),
              help='Design rules JSON file')
def index_schema(connection, rules):
    """Build RAG index from database schema."""
    
    click.echo("📚 Building RAG index...")
    
    container = DIContainer()
    container.configure(connection, "postgresql")
    
    # Get services
    inspector = container.get_inspector()
    embedding_service = container.get_embedding_service()
    vector_store = container.get_vector_store()
    
    # Introspect schema
    schema = inspector.introspect_schema()
    
    # Generate embeddings
    results = embedding_service.embed_schema_objects(schema)
    
    # Add to vector store
    texts = [r[0] for r in results]
    embeddings = np.array([r[1] for r in results])
    metadata = [r[2] for r in results]
    
    vector_store.add_embeddings(texts, embeddings, metadata)
    
    # Save index
    vector_store.save("./data/schema_index")
    
    click.echo(f"✅ Index built successfully")
    click.echo(f"   Indexed objects: {len(results)}")


@cli.command()
@click.option('--port', '-p', default=8000, help='Port to run API server')
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind')
def serve(port, host):
    """Start the REST API server."""
    
    click.echo(f"🌐 Starting API server on {host}:{port}")
    
    from src.presentation.api.app import create_app
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    cli()


# ==============================================================================
# FILE: src/presentation/api/app.py
# ==============================================================================


# ==============================================================================
# FILE: src/infrastructure/di_container.py
# ==============================================================================


# ==============================================================================
# FILE: src/infrastructure/repositories/schema_repository.py
# ==============================================================================


# ==============================================================================
# FILE: src/infrastructure/repositories/uschema_repository.py
# ==============================================================================


# ==============================================================================
# FILE: src/infrastructure/repositories/rule_repository.py
# ==============================================================================


# ==============================================================================
# FILE: tests/test_diff_engine.py
# ==============================================================================


# ==============================================================================
# FILE: tests/test_integration.py
# ==============================================================================


# ==============================================================================
# FILE: pyproject.toml
# ==============================================================================


# ==============================================================================
# FILE: README.md
# ==============================================================================

# ==============================================================================
# FILE: .env.example
# ==============================================================================
