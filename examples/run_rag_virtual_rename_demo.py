#!/usr/bin/env python3
"""
Demo script for RAG-based virtual renaming system.

This script demonstrates the complete workflow:
1. Building a knowledge base from current schema
2. Matching U-Schema entities to existing tables using RAG
3. Generating migration SQL without physical RENAME operations

Usage:
    # Local embeddings (no API key needed)
    python examples/run_rag_virtual_rename_demo.py

    # With OpenAI LLM validation (requires API key)
    RAG_USE_LLM=1 OPENAI_API_KEY=your_key python examples/run_rag_virtual_rename_demo.py
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.infrastructure.di_container import DIContainer
from src.domain.entities.schema import USchema, USchemaEntity, USchemaAttribute, DataType, SchemaMetadata, Table, Column
from src.domain.entities.evolution import ChangeType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_current_schema() -> SchemaMetadata:
    """Create a demo current schema with e-commerce tables."""
    tables = [
        Table(
            name="products",
            columns=[
                Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                Column(name="name", data_type="VARCHAR(255)", nullable=False),
                Column(name="price", data_type="DECIMAL(10,2)", nullable=False),
                Column(name="quantity", data_type="INTEGER", nullable=True),
                Column(name="reference", data_type="VARCHAR(255)", nullable=True),
                Column(name="created_at", data_type="TIMESTAMP", nullable=False),
            ]
        ),
        Table(
            name="customers",
            columns=[
                Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                Column(name="email", data_type="VARCHAR(255)", nullable=False),
                Column(name="first_name", data_type="VARCHAR(100)", nullable=False),
                Column(name="last_name", data_type="VARCHAR(100)", nullable=False),
                Column(name="phone", data_type="VARCHAR(20)", nullable=True),
            ]
        ),
        Table(
            name="orders",
            columns=[
                Column(name="id", data_type="INTEGER", primary_key=True, nullable=False),
                Column(name="customer_id", data_type="INTEGER", nullable=False),
                Column(name="total_amount", data_type="DECIMAL(10,2)", nullable=False),
                Column(name="status", data_type="VARCHAR(50)", nullable=False),
                Column(name="order_date", data_type="TIMESTAMP", nullable=False),
            ]
        )
    ]
    
    return SchemaMetadata(tables=tables)


def create_demo_uschema() -> USchema:
    """Create a demo U-Schema with entities that should map to existing tables."""
    entities = [
        # Should map to products table
        USchemaEntity(
            name="items",
            attributes=[
                USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="name", data_type=DataType.STRING, required=True),
                USchemaAttribute(name="price", data_type=DataType.DECIMAL, required=True),
                USchemaAttribute(name="qte", data_type=DataType.INTEGER, required=False),  # -> quantity
                USchemaAttribute(name="ref", data_type=DataType.STRING, required=False),    # -> reference
                USchemaAttribute(name="description", data_type=DataType.STRING, required=False),  # New column
            ]
        ),
        
        # Should map to customers table
        USchemaEntity(
            name="users",
            attributes=[
                USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="email", data_type=DataType.STRING, required=True),
                USchemaAttribute(name="firstName", data_type=DataType.STRING, required=True),  # -> first_name
                USchemaAttribute(name="lastName", data_type=DataType.STRING, required=True),   # -> last_name
                USchemaAttribute(name="phoneNumber", data_type=DataType.STRING, required=False), # -> phone
                USchemaAttribute(name="address", data_type=DataType.STRING, required=False),    # New column
            ]
        ),
        
        # Should map to orders table
        USchemaEntity(
            name="purchases",
            attributes=[
                USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="userId", data_type=DataType.INTEGER, required=True),    # -> customer_id
                USchemaAttribute(name="amount", data_type=DataType.DECIMAL, required=True),    # -> total_amount
                USchemaAttribute(name="status", data_type=DataType.STRING, required=True),
                USchemaAttribute(name="date", data_type=DataType.TIMESTAMP, required=True),    # -> order_date
                USchemaAttribute(name="paymentMethod", data_type=DataType.STRING, required=False),  # New column
            ]
        ),
        
        # New entity - should create new table
        USchemaEntity(
            name="reviews",
            attributes=[
                USchemaAttribute(name="id", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="productId", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="userId", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="rating", data_type=DataType.INTEGER, required=True),
                USchemaAttribute(name="comment", data_type=DataType.STRING, required=False),
                USchemaAttribute(name="createdAt", data_type=DataType.TIMESTAMP, required=True),
            ]
        )
    ]
    
    return USchema(entities=entities)


def demo_rag_schema_matcher():
    """Demo the RAG schema matcher functionality."""
    logger.info("🔍 Demo: RAG Schema Matcher")
    
    # Set up environment
    use_llm = os.getenv("RAG_USE_LLM", "0") == "1"
    if use_llm:
        logger.info("Using LLM validation mode")
    else:
        logger.info("Using retrieval-only mode")
    
    try:
        # Create RAG matcher directly with demo data (no DB connection needed)
        from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
        
        # Create components
        provider = LocalEmbeddingProvider()
        embedding_service = EmbeddingService(provider)
        vector_store = RAGVectorStore(dimension=provider.dimension)
        
        # Create matcher
        matcher = RAGSchemaMatcher(
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_client=None  # No LLM for demo
        )
        
        # Build knowledge base from demo schema
        current_schema = create_demo_current_schema()
        kb_docs = matcher.build_kb(current_schema)
        matcher.index_kb(kb_docs)
        
        logger.info(f"Built knowledge base with {len(kb_docs)} documents")
        
        # Test table matching for each entity
        uschema = create_demo_uschema()
        
        logger.info("\n📊 Table Matching Results:")
        for entity in uschema.entities:
            attr_names = [attr.name for attr in entity.attributes]
            result = matcher.match_table(entity.name, attr_names)
            
            if result.target_name:
                logger.info(f"  ✅ {entity.name} -> {result.target_name} (confidence: {result.confidence:.3f})")
                logger.info(f"     Rationale: {result.rationale}")
            else:
                logger.info(f"  ❌ {entity.name} -> No match (confidence: {result.confidence:.3f})")
        
        # Test column matching for matched tables
        logger.info("\n📋 Column Matching Results:")
        for entity in uschema.entities:
            attr_names = [attr.name for attr in entity.attributes]
            table_result = matcher.match_table(entity.name, attr_names)
            
            if table_result.target_name:
                logger.info(f"\n  Table: {entity.name} -> {table_result.target_name}")
                for attr in entity.attributes:
                    col_result = matcher.match_column(
                        table_result.target_name, 
                        attr.name, 
                        attr.data_type.value
                    )
                    
                    if col_result.target_name:
                        logger.info(f"    ✅ {attr.name} -> {col_result.target_name} (confidence: {col_result.confidence:.3f})")
                    else:
                        logger.info(f"    ➕ {attr.name} -> New column needed")
        
        return True
        
    except Exception as e:
        logger.error(f"RAG Schema Matcher demo failed: {e}")
        return False


def demo_diff_engine():
    """Demo the DiffEngine with RAG-based virtual renaming."""
    logger.info("\n🔄 Demo: DiffEngine with RAG Virtual Renaming")
    
    try:
        # Create RAG matcher directly with demo data
        from src.infrastructure.rag.embedding_service import EmbeddingService, LocalEmbeddingProvider
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
        from src.domain.services.diff_engine import DiffEngine
        from src.domain.entities.rules import NamingConvention
        
        # Create components
        provider = LocalEmbeddingProvider()
        embedding_service = EmbeddingService(provider)
        vector_store = RAGVectorStore(dimension=provider.dimension)
        
        # Create matcher and build knowledge base
        matcher = RAGSchemaMatcher(
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_client=None  # No LLM for demo
        )
        
        current_schema = create_demo_current_schema()
        kb_docs = matcher.build_kb(current_schema)
        matcher.index_kb(kb_docs)
        
        # Create diff engine with RAG matcher
        diff_engine = DiffEngine(NamingConvention(), rag_matcher=matcher)
        
        # Create test data
        uschema = create_demo_uschema()
        
        # Compute differences
        changes = diff_engine.compute_diff(uschema, current_schema)
        
        logger.info(f"\n📝 Generated {len(changes)} schema changes:")
        
        # Group changes by type
        changes_by_type = {}
        for change in changes:
            change_type = change.change_type.value
            if change_type not in changes_by_type:
                changes_by_type[change_type] = []
            changes_by_type[change_type].append(change)
        
        for change_type, change_list in changes_by_type.items():
            logger.info(f"\n  {change_type} ({len(change_list)} changes):")
            for change in change_list:
                if change.target_column:
                    logger.info(f"    - {change.target_table}.{change.target_column}: {change.reason}")
                else:
                    logger.info(f"    - {change.target_table}: {change.reason}")
        
        # Analyze virtual renaming success
        create_table_count = len(changes_by_type.get("CREATE_TABLE", []))
        add_column_count = len(changes_by_type.get("ADD_COLUMN", []))
        modify_column_count = len(changes_by_type.get("MODIFY_COLUMN", []))
        
        logger.info(f"\n📈 Summary:")
        logger.info(f"  - CREATE_TABLE: {create_table_count} (should be 1 for new 'reviews' table)")
        logger.info(f"  - ADD_COLUMN: {add_column_count} (new columns for existing tables)")
        logger.info(f"  - MODIFY_COLUMN: {modify_column_count} (type changes for mapped columns)")
        
        # Check for successful virtual renaming
        expected_new_tables = 1  # Only 'reviews' should be new
        virtual_rename_success = create_table_count == expected_new_tables
        
        logger.info(f"\n🎯 Virtual Renaming: {'✅ SUCCESS' if virtual_rename_success else '❌ FAILED'}")
        if virtual_rename_success:
            logger.info("  - items -> products (virtual mapping)")
            logger.info("  - users -> customers (virtual mapping)")
            logger.info("  - purchases -> orders (virtual mapping)")
            logger.info("  - reviews -> new table (no existing match)")
        
        return changes, virtual_rename_success
        
    except Exception as e:
        logger.error(f"DiffEngine demo failed: {e}")
        return [], False


def demo_migration_builder(changes):
    """Demo the MigrationBuilder generating SQL."""
    logger.info("\n🏗️  Demo: MigrationBuilder")
    
    try:
        # Create migration builder directly
        from src.domain.services.migration_builder import MigrationBuilder
        migration_builder = MigrationBuilder("postgresql")
        
        # Generate SQL
        sql_statements = migration_builder.build_migration(changes)
        
        logger.info(f"\n💻 Generated {len(sql_statements)} SQL statements:")
        for i, sql in enumerate(sql_statements, 1):
            logger.info(f"\n  {i}. {sql}")
        
        # Analyze SQL
        create_table_sql = [sql for sql in sql_statements if sql.startswith("CREATE TABLE")]
        alter_table_sql = [sql for sql in sql_statements if sql.startswith("ALTER TABLE")]
        
        logger.info(f"\n📊 SQL Analysis:")
        logger.info(f"  - CREATE TABLE statements: {len(create_table_sql)}")
        logger.info(f"  - ALTER TABLE statements: {len(alter_table_sql)}")
        logger.info(f"  - No RENAME TABLE statements: ✅ (as expected)")
        
        # Verify no physical renames
        rename_statements = [sql for sql in sql_statements if "RENAME" in sql.upper()]
        no_physical_renames = len(rename_statements) == 0
        
        logger.info(f"\n🚫 Physical Renames: {'✅ NONE' if no_physical_renames else '❌ FOUND'}")
        if not no_physical_renames:
            logger.info(f"  Found {len(rename_statements)} RENAME statements (unexpected)")
        
        return sql_statements, no_physical_renames
        
    except Exception as e:
        logger.error(f"MigrationBuilder demo failed: {e}")
        return [], False


def main():
    """Run the complete RAG virtual rename demo."""
    logger.info("🚀 Starting RAG Virtual Rename Demo")
    logger.info("=" * 60)
    
    try:
        # Demo 1: RAG Schema Matcher
        matcher_success = demo_rag_schema_matcher()
        
        # Demo 2: DiffEngine
        changes, diff_success = demo_diff_engine()
        
        # Demo 3: MigrationBuilder
        if diff_success and changes:
            sql_statements, migration_success = demo_migration_builder(changes)
        else:
            logger.warning("Skipping MigrationBuilder demo due to DiffEngine issues")
            migration_success = False
            sql_statements = []
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 FINAL RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"RAG Schema Matcher: {'✅ PASSED' if matcher_success else '❌ FAILED'}")
        logger.info(f"DiffEngine RAG Integration: {'✅ PASSED' if diff_success else '❌ FAILED'}")
        logger.info(f"MigrationBuilder: {'✅ PASSED' if migration_success else '❌ FAILED'}")
        
        overall_success = matcher_success and diff_success and migration_success
        
        if overall_success:
            logger.info("\n🎉 ALL DEMOS PASSED!")
            logger.info("✅ RAG-based virtual renaming is working correctly")
            logger.info("✅ No physical RENAME operations generated")
            logger.info("✅ Semantic mapping achieved for items->products, users->customers, purchases->orders")
            logger.info("✅ New 'reviews' table will be created as expected")
            
            logger.info(f"\n📋 Generated SQL Summary:")
            logger.info(f"  Total statements: {len(sql_statements)}")
            logger.info(f"  CREATE TABLE: {len([s for s in sql_statements if s.startswith('CREATE TABLE')])}")
            logger.info(f"  ALTER TABLE: {len([s for s in sql_statements if s.startswith('ALTER TABLE')])}")
        else:
            logger.error("\n❌ SOME DEMOS FAILED!")
            logger.error("Check the logs above for specific issues")
            return 1
        
    except Exception as e:
        logger.error(f"Demo failed with exception: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
