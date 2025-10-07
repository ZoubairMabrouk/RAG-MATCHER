"""FastAPI application."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid

from src.application.dtos.evolution_dto import EvolutionRequest
from src.infrastructure.di_container import DIContainer


# Pydantic models for API
class USchemaInput(BaseModel):
    """Input model for U-Schema."""
    uschema: Dict[str, Any]
    connection_string: str
    use_rag: bool = True
    dry_run: bool = True
    dialect: str = "postgresql"


class EvolutionPlanOutput(BaseModel):
    """Output model for evolution plan."""
    job_id: str
    description: str
    risk_level: str
    changes_count: int
    sql_statements: List[str]
    estimated_duration_minutes: int
    validation_results: Dict[str, Any]


class IntrospectionOutput(BaseModel):
    """Output model for schema introspection."""
    database_name: str
    tables_count: int
    tables: List[Dict[str, Any]]


# Job storage (in production, use Redis or database)
jobs_store = {}


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Database Evolution System",
        description="LLM-powered schema migration tool with RAG",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def root():
        """Root endpoint."""
        return {
            "name": "Database Evolution System",
            "version": "1.0.0",
            "endpoints": {
                "analyze": "/api/v1/analyze",
                "introspect": "/api/v1/introspect",
                "jobs": "/api/v1/jobs/{job_id}"
            }
        }
    
    @app.post("/api/v1/analyze", response_model=EvolutionPlanOutput)
    async def analyze_evolution(input_data: USchemaInput, background_tasks: BackgroundTasks):
        """
        Analyze schema evolution and generate plan.
        """
        try:
            # Create job ID
            job_id = str(uuid.uuid4())
            
            # Create request
            request = EvolutionRequest(
                uschema_json=input_data.uschema,
                database_connection=input_data.connection_string,
                include_rag=input_data.use_rag,
                dry_run=input_data.dry_run,
                safe_mode=True,
                target_dialect=input_data.dialect
            )
            
            # Configure DI container
            container = DIContainer()
            container.configure(input_data.connection_string, input_data.dialect)
            orchestrator = container.get_orchestrator()
            
            # Process evolution
            response = orchestrator.process_evolution(request)
            
            # Store job
            jobs_store[job_id] = {
                "status": "completed",
                "response": response
            }
            
            return EvolutionPlanOutput(
                job_id=job_id,
                description=response.plan.description,
                risk_level=response.plan.risk_level,
                changes_count=len(response.plan.changes),
                sql_statements=response.sql_statements,
                estimated_duration_minutes=response.estimated_duration_minutes,
                validation_results=response.validation_results
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/v1/introspect", response_model=IntrospectionOutput)
    async def introspect_schema(connection_string: str):
        """
        Introspect database schema.
        """
        try:
            from src.infrastructure.database.inspector import PostgresInspector
            
            inspector = PostgresInspector(connection_string)
            schema = inspector.introspect_schema()
            
            tables_data = [
                {
                    "name": t.name,
                    "schema": t.schema,
                    "columns_count": len(t.columns),
                    "row_count": t.row_count
                }
                for t in schema.tables
            ]
            
            return IntrospectionOutput(
                database_name=schema.database_name,
                tables_count=len(schema.tables),
                tables=tables_data
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """
        Get job status and results.
        """
        if job_id not in jobs_store:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return jobs_store[job_id]
    
    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app
