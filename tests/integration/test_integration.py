"""Integration tests."""

import unittest
from src.application.dtos.evolution_dto import EvolutionRequest
from src.infrastructure.di_container import DIContainer


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @unittest.skip("Requires database connection")
    def test_end_to_end_evolution(self):
        """Test complete evolution process."""
        # Arrange
        uschema_json = {
            "entities": [
                {
                    "name": "Product",
                    "attributes": [
                        {"name": "id", "type": "uuid", "required": True},
                        {"name": "name", "type": "string", "required": True},
                        {"name": "price", "type": "decimal", "required": True}
                    ]
                }
            ]
        }
        
        request = EvolutionRequest(
            uschema_json=uschema_json,
            database_connection="postgresql://localhost/testdb",
            include_rag=False,
            dry_run=True
        )
        
        # Act
        container = DIContainer()
        container.configure(request.database_connection, "postgresql")
        orchestrator = container.get_orchestrator()
        
        response = orchestrator.process_evolution(request)
        
        # Assert
        self.assertIsNotNone(response.plan)
        self.assertGreater(len(response.sql_statements), 0)
