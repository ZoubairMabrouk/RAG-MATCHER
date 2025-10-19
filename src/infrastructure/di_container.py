"""Dependency Injection Container."""

from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class DIContainer:
    """
    Dependency Injection Container.
    Follows Dependency Inversion Principle.
    """
    
    def __init__(self):
        self._connection_string: Optional[str] = None
        self._dialect: str = "postgresql"
        self._services = {}
    
    def configure(self, connection_string: str, dialect: str = "postgresql"):
        """Configure the container."""
        self._connection_string = connection_string
        self._dialect = dialect
    
    def get_inspector(self):
        """Get database inspector."""
        if "inspector" not in self._services:
            from src.infrastructure.database.inspector import PostgresInspector
            self._services["inspector"] = PostgresInspector(self._connection_string)
        return self._services["inspector"]
    
    def get_embedding_service(self):
        """Get embedding service."""
        if "embedding_service" not in self._services:
            from src.infrastructure.rag.embedding_service import (
                EmbeddingService, OpenAIEmbeddingProvider
            )
            
            api_key = os.getenv("OPENAI_API_KEY", "")
            provider = OpenAIEmbeddingProvider(api_key)
            self._services["embedding_service"] = EmbeddingService(provider)
        
        return self._services["embedding_service"]
    
    def get_vector_store(self):
        """Get vector store."""
        if "vector_store" not in self._services:
            from src.infrastructure.rag.vector_store import RAGVectorStore
            # Get dimension from embedding service
            embedding_service = self.get_embedding_service()
            dimension = embedding_service.dimension
            self._services["vector_store"] = RAGVectorStore(dimension=dimension, index_type='auto')
        
        return self._services["vector_store"]
    
    def get_llm_client(self):
        """Get LLM client."""
        if "llm_client" not in self._services:
            from src.infrastructure.llm.llm_client import OpenAILLMClient
            
            api_key = os.getenv("OPENAI_API_KEY", "")
            self._services["llm_client"] = OpenAILLMClient(api_key)
        
        return self._services["llm_client"]
    
    def get_rag_schema_matcher(self):
        """Get RAG schema matcher."""
        if "rag_schema_matcher" not in self._services:
            from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
            
            # Get dependencies
            embedding_service = self.get_embedding_service()
            vector_store = self.get_vector_store()
            
            # Check if LLM should be used
            use_llm = os.getenv("RAG_USE_LLM", "0") == "1"
            llm_client = self.get_llm_client() if use_llm else None
            
            # Create matcher
            matcher = RAGSchemaMatcher(
                embedding_service=embedding_service,
                vector_store=vector_store,
                llm_client=llm_client
            )
            
            # Build knowledge base from current schema
            try:
                inspector = self.get_inspector()
                schema = inspector.introspect_schema()
                kb_docs = matcher.build_kb(schema)
                matcher.index_kb(kb_docs)
                logger.info(f"[DIContainer] Built RAG knowledge base with {len(kb_docs)} documents")
            except Exception as e:
                logger.warning(f"[DIContainer] Failed to build RAG knowledge base: {e}")
            
            self._services["rag_schema_matcher"] = matcher
        
        return self._services["rag_schema_matcher"]
    
    def get_diff_engine(self):
        """Get diff engine."""
        if "diff_engine" not in self._services:
            from src.domain.services.diff_engine import DiffEngine
            from src.domain.entities.rules import NamingConvention
            
            naming = NamingConvention()
            matcher = self.get_rag_schema_matcher()
            self._services["diff_engine"] = DiffEngine(naming, rag_matcher=matcher)
        
        return self._services["diff_engine"]
    
    def get_rule_engine(self):
        """Get rule engine."""
        if "rule_engine" not in self._services:
            from src.domain.services.rule_engine import RuleEngine
            from src.domain.entities.rules import RuleSet, NamingConvention
            
            rules = RuleSet(naming=NamingConvention(), design_rules=[])
            self._services["rule_engine"] = RuleEngine(rules)
        
        return self._services["rule_engine"]
    
    def get_migration_builder(self):
        """Get migration builder."""
        if "migration_builder" not in self._services:
            from src.domain.services.migration_builder import MigrationBuilder
            self._services["migration_builder"] = MigrationBuilder(self._dialect)
        
        return self._services["migration_builder"]
    
    def get_orchestrator(self):
        """Get evolution orchestrator."""
        from src.application.orchestrators.evolution_orchestrator import EvolutionOrchestrator
        from src.application.use_case.analyze_evolution import AnalyzeEvolutionUseCase
        from src.application.use_case.generate_migration import GenerateMigrationUseCase
        from src.infrastructure.validators.sql_validator import SQLValidator, SafetyValidator
        from src.infrastructure.rag.retriever import RAGRetriever
        
        # Build use cases
        rag_retriever = RAGRetriever(
            self.get_vector_store(),
            self.get_embedding_service()
        )
        
        # Repositories (simplified - would use proper implementations)
        from src.infrastructure.repositories.schema_repository import SchemaRepository
        from src.infrastructure.repositories.uschema_repository import USchemaRepository
        from src.infrastructure.repositories.rule_repository import RuleRepository
        
        schema_repo = SchemaRepository(self.get_inspector())
        uschema_repo = USchemaRepository()
        rule_repo = RuleRepository()
        
        analyze_use_case = AnalyzeEvolutionUseCase(
            schema_repo,
            uschema_repo,
            rule_repo,
            self.get_diff_engine(),
            self.get_rule_engine(),
            rag_retriever
        )
        
        sql_validator = SQLValidator(self._dialect)
        safety_validator = SafetyValidator(self.get_inspector())
        
        generate_use_case = GenerateMigrationUseCase(
            self.get_llm_client(),
            self.get_migration_builder(),
            sql_validator,
            safety_validator
        )
        
        return EvolutionOrchestrator(analyze_use_case, generate_use_case)
