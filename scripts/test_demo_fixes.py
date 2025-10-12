#!/usr/bin/env python3
"""
Test script to verify demo fixes work correctly.

This script tests the specific fixes made to the demo:
1. DI Container using correct method name (introspect_schema)
2. Demo working without real DB connection
3. RAG matcher building KB from demo data
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_di_container_inspect_method():
    """Test that DI container uses correct method name."""
    logger.info("🧪 Testing DI Container inspect method...")
    
    try:
        from src.infrastructure.di_container import DIContainer
        
        # Create container
        container = DIContainer()
        container.configure("postgresql://test:test@localhost:5432/test")
        
        # Get inspector
        inspector = container.get_inspector()
        
        # Check that inspector has introspect_schema method
        assert hasattr(inspector, 'introspect_schema'), "Inspector should have introspect_schema method"
        assert not hasattr(inspector, 'inspect'), "Inspector should NOT have inspect method (deprecated)"
        
        logger.info("✅ DI Container inspect method test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ DI Container inspect method test failed: {e}")
        return False


def test_rag_matcher_with_demo_data():
    """Test RAG matcher with demo data (no DB connection)."""
    logger.info("🧪 Testing RAG Matcher with demo data...")
    
    try:
        from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
        from src.domain.entities.schema import SchemaMetadata, Table, Column
        
        # Create demo schema
        demo_schema = SchemaMetadata(tables=[
            Table(
                name="products",
                columns=[
                    Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                    Column(name="name", data_type="VARCHAR(255)", nullable=False),
                    Column(name="price", data_type="DECIMAL(10,2)", nullable=False),
                    Column(name="quantity", data_type="INTEGER", nullable=True),
                ]
            )
        ])
        
        # Create components
        provider = LocalEmbeddingProvider()
        embedding_service = EmbeddingService(provider)
        vector_store = RAGVectorStore(dimension=provider.dimension, force_flat=True)
        
        # Create matcher
        matcher = RAGSchemaMatcher(
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Build knowledge base from demo data
        kb_docs = matcher.build_kb(demo_schema)
        matcher.index_kb(kb_docs)
        
        # Verify KB was built
        assert len(kb_docs) > 0, "Should have created KB documents"
        assert matcher._kb_built, "Knowledge base should be marked as built"
        
        # Test table matching
        result = matcher.match_table("items", ["id", "name", "price", "quantity"])
        
        # Should find a match with products table
        assert result is not None, "Should return a match result"
        assert result.target_name == "products", f"Expected 'products', got '{result.target_name}'"
        assert result.confidence > 0.5, f"Confidence should be > 0.5, got {result.confidence}"
        
        logger.info("✅ RAG Matcher with demo data test passed")
        logger.info(f"   KB documents: {len(kb_docs)}")
        logger.info(f"   Table match: {result.target_name} (confidence: {result.confidence:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG Matcher with demo data test failed: {e}")
        return False


def test_diff_engine_with_rag_matcher():
    """Test DiffEngine with RAG matcher using demo data."""
    logger.info("🧪 Testing DiffEngine with RAG matcher...")
    
    try:
        from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
        from src.domain.services.diff_engine import DiffEngine
        from src.domain.entities.rules import NamingConvention
        from src.domain.entities.schema import SchemaMetadata, Table, Column, USchema, USchemaEntity, USchemaAttribute, DataType
        from src.domain.entities.evolution import ChangeType
        
        # Create demo schemas
        current_schema = SchemaMetadata(tables=[
            Table(
                name="products",
                columns=[
                    Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                    Column(name="name", data_type="VARCHAR(255)", nullable=False),
                    Column(name="price", data_type="DECIMAL(10,2)", nullable=False),
                    Column(name="quantity", data_type="INTEGER", nullable=True),
                ]
            )
        ])
        
        uschema = USchema(entities=[
            USchemaEntity(
                name="items",
                attributes=[
                    USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
                    USchemaAttribute(name="name", data_type=DataType.STRING, required=True),
                    USchemaAttribute(name="price", data_type=DataType.DECIMAL, required=True),
                    USchemaAttribute(name="qte", data_type=DataType.INTEGER, required=False),  # -> quantity
                ]
            )
        ])
        
        # Create RAG matcher
        provider = LocalEmbeddingProvider()
        embedding_service = EmbeddingService(provider)
        vector_store = RAGVectorStore(dimension=provider.dimension)
        
        matcher = RAGSchemaMatcher(
            embedding_service=embedding_service,
            vector_store=vector_store
        )
        
        # Build knowledge base
        kb_docs = matcher.build_kb(current_schema)
        matcher.index_kb(kb_docs)
        
        # Create diff engine with RAG matcher
        diff_engine = DiffEngine(NamingConvention(), rag_matcher=matcher)
        
        # Compute differences
        changes = diff_engine.compute_diff(uschema, current_schema)
        
        # Verify virtual renaming worked
        create_table_changes = [c for c in changes if c.change_type == ChangeType.CREATE_TABLE]
        add_column_changes = [c for c in changes if c.change_type == ChangeType.ADD_COLUMN]
        
        # Should NOT create new table (items should map to products)
        assert len(create_table_changes) == 0, f"Should not create new table, got {len(create_table_changes)} CREATE_TABLE changes"
        
        # Should add new columns to existing products table
        assert len(add_column_changes) > 0, "Should add new columns to existing table"
        
        # Check that changes are for products table
        for change in add_column_changes:
            assert change.target_table == "products", f"Changes should target products table, got {change.target_table}"
        
        logger.info("✅ DiffEngine with RAG matcher test passed")
        logger.info(f"   CREATE_TABLE changes: {len(create_table_changes)}")
        logger.info(f"   ADD_COLUMN changes: {len(add_column_changes)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ DiffEngine with RAG matcher test failed: {e}")
        return False


def main():
    """Run all demo fix tests."""
    logger.info("🚀 Starting Demo Fix Tests")
    logger.info("=" * 60)
    
    tests = [
        ("DI Container Inspect Method", test_di_container_inspect_method),
        ("RAG Matcher with Demo Data", test_rag_matcher_with_demo_data),
        ("DiffEngine with RAG Matcher", test_diff_engine_with_rag_matcher),
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
    logger.info("🎯 DEMO FIX TEST SUMMARY")
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
        logger.info("🎉 ALL DEMO FIXES WORKING!")
        logger.info("✅ Demo should now work without DB connection")
        logger.info("✅ RAG virtual renaming should function correctly")
        return 0
    else:
        logger.error("❌ SOME DEMO FIXES NEED ATTENTION!")
        logger.error("Please review the failed tests above")
        return 1


if __name__ == "__main__":
    exit(main())
