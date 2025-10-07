"""Main orchestrator for evolution process."""
from typing import List

from src.application.dtos.evolution_dto import EvolutionResponse, EvolutionRequest
from src.application.use_case.analyze_evolution import AnalyzeEvolutionUseCase
from src.application.use_case.generate_migration import GenerateMigrationUseCase

class EvolutionOrchestrator:
    """
    Main orchestrator coordinating the entire evolution process.
    Single Responsibility: Coordinate use cases and present results.
    """
    
    def __init__(
        self,
        analyze_use_case: AnalyzeEvolutionUseCase,
        generate_use_case: GenerateMigrationUseCase
    ):
        self._analyze = analyze_use_case
        self._generate = generate_use_case
    
    def process_evolution(self, request: EvolutionRequest) -> EvolutionResponse:
        """Process complete evolution request."""
        
        # Step 1: Analyze
        print("🔍 Analyzing schema differences...")
        analysis = self._analyze.execute(request)
        
        print(f"   Found {len(analysis['changes'])} potential changes")
        
        # Step 2: Generate migration
        print("🛠️  Generating migration plan...")
        plan = self._generate.execute(analysis, use_llm=request.include_rag)
        
        # Step 3: Build SQL
        print("📝 Building SQL statements...")
        from src.domain.services.migration_builder import MigrationBuilder
        builder = MigrationBuilder(dialect=request.target_dialect)
        sql_statements = builder.build_migration(plan.changes)
        
        # Step 4: Prepare response
        response = EvolutionResponse(
            plan=plan,
            sql_statements=sql_statements,
            validation_results=plan.validation_results,
            estimated_duration_minutes=plan.estimated_duration_minutes
        )
        
        # Step 5: Execute if not dry run
        if not request.dry_run and plan.validation_results.get("sql_valid"):
            print("⚡ Executing migration...")
            execution_report = self._execute_migration(
                sql_statements,
                request.database_connection,
                safe_mode=request.safe_mode
            )
            response.execution_report = execution_report
        else:
            print("✅ Dry run complete - no changes applied")
        
        return response
    
    def _execute_migration(
        self, 
        sql_statements: List[str],
        connection_string: str,
        safe_mode: bool
    ) -> str:
        """Execute migration statements."""
        import psycopg2
        
        conn = psycopg2.connect(connection_string)
        report_lines = []
        
        try:
            with conn.cursor() as cur:
                if safe_mode:
                    # Execute in transaction
                    cur.execute("BEGIN")
                
                for i, sql in enumerate(sql_statements):
                    try:
                        cur.execute(sql)
                        report_lines.append(f"✓ Statement {i+1} executed successfully")
                    except Exception as e:
                        report_lines.append(f"✗ Statement {i+1} failed: {e}")
                        if safe_mode:
                            cur.execute("ROLLBACK")
                            report_lines.append("⚠️  Transaction rolled back")
                            break
                
                if safe_mode:
                    cur.execute("COMMIT")
                    report_lines.append("✓ Transaction committed")
            
            return "\n".join(report_lines)
            
        finally:
            conn.close()
