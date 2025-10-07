
from src.domain.repositeries.interfaces import IRuleRepository, ISchemaRepository, IUSchemaRepository
from src.domain.services.diff_engine import DiffEngine
from src.domain.services.rule_engine import RuleEngine
from src.infrastructure.rag.retriever import RAGRetriever
from src.application.dtos.evolution_dto import EvolutionRequest
from typing import Dict, Any
class AnalyzeEvolutionUseCase:
    """
    Use case: Analyze required evolution between U-Schema and current DB.
    Single Responsibility: Orchestrate evolution analysis.
    """
    
    def __init__(
        self,
        schema_repository: ISchemaRepository,
        uschema_repository: IUSchemaRepository,
        rule_repository: IRuleRepository,
        diff_engine: DiffEngine,
        rule_engine: RuleEngine,
        rag_retriever: RAGRetriever
    ):
        self._schema_repo = schema_repository
        self._uschema_repo = uschema_repository
        self._rule_repo = rule_repository
        self._diff_engine = diff_engine
        self._rule_engine = rule_engine
        self._rag_retriever = rag_retriever
    
    def execute(self, request: EvolutionRequest) -> Dict[str, Any]:
        """Execute the analysis."""
        # 1. Parse U-Schema
        uschema = self._uschema_repo.parse_uschema(request.uschema_json)
        
        # 2. Get current schema
        current_schema = self._schema_repo.get_current_schema()
        
        # 3. Compute symbolic diff
        changes = self._diff_engine.compute_diff(uschema, current_schema)
        
        # 4. Retrieve relevant context via RAG
        rag_context = {}
        if request.include_rag:
            entity_names = [e.name for e in uschema.entities]
            relevant_tables = self._rag_retriever.retrieve_relevant_tables(entity_names)
            relevant_rules = self._rag_retriever.retrieve_design_rules("schema evolution")
            
            rag_context = {
                "tables": relevant_tables,
                "rules": relevant_rules
            }
        
        # 5. Validate against rules
        rule_violations = self._rule_engine.validate_changes(changes)
        
        # 6. Enrich with best practices
        enriched_changes = self._rule_engine.enrich_changes_with_best_practices(changes)
        
        return {
            "uschema": uschema,
            "current_schema": current_schema,
            "changes": enriched_changes,
            "rag_context": rag_context,
            "rule_violations": rule_violations
        }
