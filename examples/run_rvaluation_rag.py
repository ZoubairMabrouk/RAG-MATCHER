# examples/run_evolution_real.py
import os
import sys
from pathlib import Path

# Ensure "src" is importable when running from project root
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.application.dtos.evolution_dto import EvolutionRequest
from src.infrastructure.di_container import DIContainer

def main():
    # 1) DB connection (use your persistent dev DB)
    db_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:55432/test")

    # 2) Minimal U-Schema example (adjust to your real input)
    uschema_json = {
        "entities": [
            {
                "name": "Product",
                "attributes": [
                    {"name": "id", "type": "uuid", "required": True},
                    {"name": "name", "type": "string", "required": True},
                    {"name": "price", "type": "decimal", "required": True},
                ],
            }
        ]
    }

    # 3) Build orchestrator from your DI container
    container = DIContainer()
    container.configure(db_url, "postgresql")
    orchestrator = container.get_orchestrator()

    # 4) Build a real request
    #    - include_rag=True will use your RAGRetriever (vector store + embeddings)
    #    - dry_run=True to preview SQL; set False to actually apply
    req = EvolutionRequest(
        uschema_json=uschema_json,
        database_connection=db_url,
        include_rag=False,   # flip to True if you want RAG in the analysis loop
        dry_run=True         # flip to False to execute generated SQL
    )

    # 5) Run the evolution orchestration
    resp = orchestrator.process_evolution(req)

    # 6) Print a clean, “real” output
    print("\n=== Evolution Plan ===")
    print(resp.plan)

    print("\n=== SQL Statements ===")
    for i, stmt in enumerate(resp.sql_statements, 1):
        print(f"{i:02d}. {stmt}")

    if getattr(resp, "warnings", None):
        print("\n=== Warnings ===")
        for w in resp.warnings:
            print(f"- {w}")

    if getattr(resp, "errors", None):
        print("\n=== Errors ===")
        for e in resp.errors:
            print(f"- {e}")

if __name__ == "__main__":
    main()
