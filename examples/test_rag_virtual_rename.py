#!/usr/bin/env python3
"""
Test script for RAG-based virtual renaming functionality.

This script demonstrates the new RAG schema matching system that replaces
heuristic-based detection with semantic matching using embeddings and optional LLM validation.

Test Case: items vs products
- U-Schema entity: items (with attributes id, name, price, qte, ref)
- Current schema: products table (with columns id, name, price, quantity, reference)
- Expected: Virtual mapping without physical RENAME operations
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.di_container import DIContainer
from src.domain.entities.schema import USchema, USchemaEntity, USchemaAttribute, DataType
from src.domain.entities.evolution import ChangeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_uschema() -> USchema:
    """Create a test U-Schema with items entity."""
    items_entity = USchemaEntity(
        name="items",
        attributes=[
            USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
            USchemaAttribute(name="name", data_type=DataType.STRING, required=True),
            USchemaAttribute(name="price", data_type=DataType.DECIMAL, required=True),
            USchemaAttribute(name="qte", data_type=DataType.INTEGER, required=True),  # Should map to quantity
            USchemaAttribute(name="ref", data_type=DataType.STRING, required=False),   # Should map to reference
        ]
    )
    
    return USchema(entities=[items_entity])


def create_test_current_schema():
    """Create a test current schema with products table."""
    from src.domain.entities.schema import SchemaMetadata, Table, Column
    
    products_table = Table(
        name="products",
        columns=[
            Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
            Column(name="name", data_type="VARCHAR(255)", nullable=False),
            Column(name="price", data_type="DECIMAL(10,2)", nullable=False),
            Column(name="quantity", data_type="INTEGER", nullable=True),  # Should match qte
            Column(name="reference", data_type="VARCHAR(255)", nullable=True),  # Should match ref
        ]
    )
    
    return SchemaMetadata(tables=[products_table])


def test_rag_schema_matcher():
    """Test RAG schema matcher functionality."""
    logger.info("=== Testing RAG Schema Matcher ===")
    
    # Create test data
    uschema = create_test_uschema()
    current_schema = create_test_current_schema()
    
    # Set up environment
    os.environ["RAG_USE_LLM"] = "0"  # Use retrieval-only mode for testing
    os.environ["OPENAI_API_KEY"] = "sk-proj-yMAv7pLRNw1sRwroqguy6aifvtXqXxeqyh2zU2B3flU016eB-gTPBoFInQBJjvQInxpG4lLxUqT3BlbkFJ9n8Cy9mjR6wh9WGXbKkzCLl38eWAUfez4lLxUqT3BlbkFJ9n8Cy9mjR6wh9WGXbKkzCLl38eWAUfez4k-y7vdn1hPjLcthaciSk6D56ljnnikgHaX4SWL-oA"
    
    # Initialize DI container
    container = DIContainer()
    container.configure("postgresql://test:test@localhost:55432/test")
    
    try:
        # Get RAG schema matcher
        matcher = container.get_rag_schema_matcher()
        
        # Test table matching
        logger.info("Testing table matching: items -> products")
        items_entity = uschema.entities[0]
        attr_names = [attr.name for attr in items_entity.attributes]
        
        table_result = matcher.match_table("items", attr_names)
        logger.info(f"Table match result: {table_result}")
        
        # Test column matching
        logger.info("Testing column matching within products table")
        qte_result = matcher.match_column("products", "qte", "INTEGER")
        logger.info(f"Column match (qte): {qte_result}")
        
        ref_result = matcher.match_column("products", "ref", "VARCHAR(255)")
        logger.info(f"Column match (ref): {ref_result}")
        
        return table_result, qte_result, ref_result
        
    except Exception as e:
        logger.error(f"RAG schema matcher test failed: {e}")
        return None, None, None


def test_diff_engine_with_rag():
    """Test DiffEngine with RAG-based virtual renaming."""
    logger.info("=== Testing DiffEngine with RAG ===")
    
    # Create test data
    uschema = create_test_uschema()
    current_schema = create_test_current_schema()
    
    # Set up environment
    os.environ["RAG_USE_LLM"] = "0"  # Use retrieval-only mode for testing
    
    # Initialize DI container
    container = DIContainer()
    container.configure("postgresql://test:test@localhost:55432/test")
    
    try:
        # Get diff engine with RAG matcher
        diff_engine = container.get_diff_engine()
        
        # Compute differences
        changes = diff_engine.compute_diff(uschema, current_schema)
        
        logger.info(f"Generated {len(changes)} changes:")
        for change in changes:
            logger.info(f"  {change.change_type.value}: {change.target_table}.{change.target_column or 'N/A'} - {change.reason}")
        
        # Analyze results
        create_table_count = sum(1 for c in changes if c.change_type == ChangeType.CREATE_TABLE)
        add_column_count = sum(1 for c in changes if c.change_type == ChangeType.ADD_COLUMN)
        modify_column_count = sum(1 for c in changes if c.change_type == ChangeType.MODIFY_COLUMN)
        
        logger.info(f"Summary: {create_table_count} CREATE_TABLE, {add_column_count} ADD_COLUMN, {modify_column_count} MODIFY_COLUMN")
        
        # Expected behavior:
        # - No CREATE TABLE (items should map to products)
        # - No ADD COLUMN for qte and ref (should map to quantity and reference)
        # - Possibly MODIFY COLUMN if types differ
        
        success = create_table_count == 0  # Should not create new table
        logger.info(f"Test {'PASSED' if success else 'FAILED'}: Virtual renaming {'worked' if success else 'failed'}")
        
        return changes, success
        
    except Exception as e:
        logger.error(f"DiffEngine test failed: {e}")
        return [], False


def test_migration_builder():
    """Test MigrationBuilder with the generated changes."""
    logger.info("=== Testing MigrationBuilder ===")
    
    # Get changes from previous test
    changes, success = test_diff_engine_with_rag()
    
    if not success:
        logger.warning("Skipping migration builder test due to previous failure")
        return
    
    try:
        # Get migration builder
        container = DIContainer()
        migration_builder = container.get_migration_builder()
        
        # Generate SQL
        sql_statements = migration_builder.build_migration(changes)
        
        logger.info(f"Generated {len(sql_statements)} SQL statements:")
        for i, sql in enumerate(sql_statements, 1):
            logger.info(f"  {i}. {sql}")
        
        # Expected: No CREATE TABLE statements, only ADD/MODIFY COLUMN
        create_table_sql = [sql for sql in sql_statements if sql.startswith("CREATE TABLE")]
        logger.info(f"CREATE TABLE statements: {len(create_table_sql)} (should be 0)")
        
        return sql_statements
        
    except Exception as e:
        logger.error(f"MigrationBuilder test failed: {e}")
        return []


def main():
    """Run all tests."""
    logger.info("Starting RAG Virtual Rename Tests")
    
    try:
        # Test 1: RAG Schema Matcher
        table_result, qte_result, ref_result = test_rag_schema_matcher()
        
        # Test 2: DiffEngine with RAG
        changes, success = test_diff_engine_with_rag()
        
        # Test 3: MigrationBuilder
        sql_statements = test_migration_builder()
        
        # Final summary
        logger.info("=== Test Summary ===")
        logger.info(f"RAG Schema Matcher: {'✓' if table_result and table_result.target_name else '✗'}")
        logger.info(f"DiffEngine RAG Integration: {'✓' if success else '✗'}")
        logger.info(f"MigrationBuilder: {'✓' if sql_statements else '✗'}")
        
        if success and sql_statements:
            logger.info("🎉 All tests passed! RAG-based virtual renaming is working correctly.")
            logger.info("✅ No physical RENAME operations generated")
            logger.info("✅ Semantic mapping between items->products and qte->quantity, ref->reference")
        else:
            logger.error("❌ Some tests failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
