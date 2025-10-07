"""Unit tests for DiffEngine."""

import unittest
from src.domain.services.diff_engine import DiffEngine
from src.domain.entities.schema import *
from src.domain.entities.rules import NamingConvention


class TestDiffEngine(unittest.TestCase):
    """Test DiffEngine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = DiffEngine(NamingConvention())
    
    def test_create_table_for_new_entity(self):
        """Test that new entity generates CREATE TABLE change."""
        # Arrange
        uschema = USchema(entities=[
            USchemaEntity(
                name="Customer",
                attributes=[
                    USchemaAttribute("id", DataType.UUID, required=True),
                    USchemaAttribute("name", DataType.STRING, required=True)
                ]
            )
        ])
        current_schema = SchemaMetadata(tables=[])
        
        # Act
        changes = self.engine.compute_diff(uschema, current_schema)
        
        # Assert
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, ChangeType.CREATE_TABLE)
        self.assertEqual(changes[0].target_table, "customers")
    
    def test_add_column_for_new_attribute(self):
        """Test that new attribute generates ADD COLUMN change."""
        # Arrange
        uschema = USchema(entities=[
            USchemaEntity(
                name="Customer",
                attributes=[
                    USchemaAttribute("id", DataType.UUID, required=True),
                    USchemaAttribute("name", DataType.STRING, required=True),
                    USchemaAttribute("email", DataType.STRING, required=False)
                ]
            )
        ])
        current_schema = SchemaMetadata(tables=[
            Table(
                name="customers",
                columns=[
                    Column("id", "UUID", nullable=False, primary_key=True),
                    Column("name", "VARCHAR(255)", nullable=False)
                ]
            )
        ])
        
        # Act
        changes = self.engine.compute_diff(uschema, current_schema)
        
        # Assert
        add_column_changes = [c for c in changes if c.change_type == ChangeType.ADD_COLUMN]
        self.assertEqual(len(add_column_changes), 1)
        self.assertEqual(add_column_changes[0].target_column, "email")
    
    def test_entity_to_table_name_conversion(self):
        """Test entity name to table name conversion."""
        # Test PascalCase to snake_case
        self.assertEqual(self.engine._entity_to_table_name("Customer"), "customers")
        self.assertEqual(self.engine._entity_to_table_name("OrderItem"), "order_items")
        self.assertEqual(self.engine._entity_to_table_name("User"), "users")
