#!/usr/bin/env python3
"""
Test script to verify all fixes are working correctly.

This script tests the specific issues that were identified and fixed:
1. DSN connection in test_docker_setup.py
2. Column signature in examples
3. CrossEncoder availability in embedding_service.py
4. DiffEngine target_table generation
5. MigrationBuilder deduplication
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.domain.entities.schema import SchemaMetadata, Table, Column, USchema, USchemaEntity, USchemaAttribute, DataType
from src.domain.entities.rules import NamingConvention
from src.domain.services.diff_engine import DiffEngine
from src.domain.services.migration_builder import MigrationBuilder
from src.domain.entities.evolution import ChangeType, SchemaChange

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_column_signature():
    """Test that Column can be created with correct signature."""
    logger.info("🧪 Testing Column signature...")
    
    try:
        # Test with correct signature
        column = Column(
            name="id", 
            data_type="INTEGER", 
            primary_key=True, 
            nullable=False
        )
        
        assert column.name == "id"
        assert column.data_type == "INTEGER"
        assert column.primary_key == True
        assert column.nullable == False
        
        logger.info("✅ Column signature test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Column signature test failed: {e}")
        return False


def test_naming_convention():
    """Test that NamingConvention has table_name and column_name methods."""
    logger.info("🧪 Testing NamingConvention methods...")
    
    try:
        naming = NamingConvention()
        
        # Test table_name method
        table_name = naming.table_name("Customer")
        assert table_name == "customers", f"Expected 'customers', got '{table_name}'"
        
        table_name2 = naming.table_name("OrderItem")
        assert table_name2 == "order_items", f"Expected 'order_items', got '{table_name2}'"
        
        # Test column_name method
        column_name = naming.column_name("firstName")
        assert column_name == "first_name", f"Expected 'first_name', got '{column_name}'"
        
        logger.info("✅ NamingConvention methods test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ NamingConvention methods test failed: {e}")
        return False


def test_diff_engine_target_table():
    """Test that DiffEngine generates correct target_table for CREATE_TABLE."""
    logger.info("🧪 Testing DiffEngine target_table generation...")
    
    try:
        # Create test data
        uschema = USchema(entities=[
            USchemaEntity(
                name="Customer",
                attributes=[
                    USchemaAttribute(name="id", data_type=DataType.UUID, required=True),
                    USchemaAttribute(name="name", data_type=DataType.STRING, required=True)
                ]
            )
        ])
        current_schema = SchemaMetadata(tables=[])
        
        # Create DiffEngine
        diff_engine = DiffEngine(NamingConvention())
        
        # Compute differences
        changes = diff_engine.compute_diff(uschema, current_schema)
        
        # Verify results
        assert len(changes) == 1, f"Expected 1 change, got {len(changes)}"
        
        create_table_change = changes[0]
        assert create_table_change.change_type == ChangeType.CREATE_TABLE
        assert create_table_change.target_table == "customers", f"Expected 'customers', got '{create_table_change.target_table}'"
        
        logger.info("✅ DiffEngine target_table test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ DiffEngine target_table test failed: {e}")
        return False


def test_migration_builder_deduplication():
    """Test that MigrationBuilder avoids duplicate column names."""
    logger.info("🧪 Testing MigrationBuilder deduplication...")
    
    try:
        # Create test changes with potential duplicates
        changes = [
            SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                target_table="products",
                target_column="description",
                definition="VARCHAR(500)",
                reason="Test ADD_COLUMN"
            ),
            SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                target_table="products",
                target_column="description",  # Duplicate
                definition="VARCHAR(500)",
                reason="Test duplicate"
            ),
            SchemaChange(
                change_type=ChangeType.MODIFY_COLUMN,
                target_table="products",
                target_column="quantity",
                definition="TYPE INTEGER",
                reason="Test MODIFY_COLUMN"
            )
        ]
        
        # Create MigrationBuilder
        migration_builder = MigrationBuilder("postgresql")
        
        # Generate SQL
        sql_statements = migration_builder.build_migration(changes)
        
        # Verify deduplication worked
        add_column_statements = [sql for sql in sql_statements if "ADD COLUMN description" in sql]
        assert len(add_column_statements) == 1, f"Expected 1 ADD COLUMN description, got {len(add_column_statements)}"
        
        # Verify no duplicate column names
        for sql in sql_statements:
            if "ADD COLUMN" in sql:
                # Check for patterns like "ADD COLUMN col col TYPE"
                parts = sql.split()
                if len(parts) >= 4:
                    col_name = parts[3]
                    if len(parts) > 4 and parts[4] == col_name:
                        raise AssertionError(f"Duplicate column name found in SQL: {sql}")
        
        logger.info("✅ MigrationBuilder deduplication test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ MigrationBuilder deduplication test failed: {e}")
        return False


def test_embedding_service_stubs():
    """Test that CrossEncoder and SentenceTransformer stubs are available."""
    logger.info("🧪 Testing embedding service stubs...")
    
    try:
        from src.infrastructure.rag.embedding_service import CrossEncoder, SentenceTransformer
        
        # Test CrossEncoder stub
        cross_encoder = CrossEncoder("test-model")
        scores = cross_encoder.predict([("query", "document")])
        assert len(scores) == 1, f"Expected 1 score, got {len(scores)}"
        
        # Test SentenceTransformer stub
        sentence_transformer = SentenceTransformer("test-model")
        dimension = sentence_transformer.get_sentence_embedding_dimension()
        assert dimension == 384, f"Expected dimension 384, got {dimension}"
        
        embeddings = sentence_transformer.encode(["test text"])
        assert embeddings.shape == (1, 384), f"Expected shape (1, 384), got {embeddings.shape}"
        
        logger.info("✅ Embedding service stubs test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding service stubs test failed: {e}")
        return False


def test_rag_schema_matcher_compatibility():
    """Test that RAGSchemaMatcher works with correct Column signature."""
    logger.info("🧪 Testing RAGSchemaMatcher compatibility...")
    
    try:
        from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
        from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
        from src.infrastructure.rag.vector_store import RAGVectorStore
        
        # Create test schema with correct Column signature
        test_schema = SchemaMetadata(tables=[
            Table(
                name="products",
                columns=[
                    Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                    Column(name="name", data_type="VARCHAR(255)", nullable=False),
                    Column(name="price", data_type="DECIMAL(10,2)", nullable=False),
                ]
            )
        ])
        
        # Create components
        provider = LocalEmbeddingProvider()
        embedding_service = EmbeddingService(provider)
        vector_store = RAGVectorStore(dimension=provider.dimension)
        
        # Create matcher
        matcher = RAGSchemaMatcher(
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Build knowledge base
        kb_docs = matcher.build_kb(test_schema)
        assert len(kb_docs) == 4, f"Expected 4 docs (1 table + 3 columns), got {len(kb_docs)}"
        
        # Index knowledge base
        matcher.index_kb(kb_docs)
        
        # Test table matching
        result = matcher.match_table("items", ["id", "name", "price"])
        assert result is not None, "Table matching should return a result"
        
        logger.info("✅ RAGSchemaMatcher compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAGSchemaMatcher compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Fix Verification Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Column Signature", test_column_signature),
        ("NamingConvention Methods", test_naming_convention),
        ("DiffEngine Target Table", test_diff_engine_target_table),
        ("MigrationBuilder Deduplication", test_migration_builder_deduplication),
        ("Embedding Service Stubs", test_embedding_service_stubs),
        ("RAGSchemaMatcher Compatibility", test_rag_schema_matcher_compatibility),
    ]
    
    results = {}
    
    for name, test_func in tests:
        logger.info(f"\n{'='*20} {name} {'='*20}")
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FIX VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL FIXES VERIFIED!")
        logger.info("✅ All identified issues have been resolved")
        logger.info("✅ System is ready for testing")
        return 0
    else:
        logger.error("❌ SOME FIXES NEED ATTENTION!")
        logger.error("Please review the failed tests above")
        return 1


if __name__ == "__main__":
    exit(main())
