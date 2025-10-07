
import json
from datetime import datetime
from typing import Dict, Any, List

from src.domain.entities.schema import *
from src.domain.entities.rules import *

from src.domain.entities.evolution import *


class TestDataFactory:
    """Factory for creating test data objects."""
    
    @staticmethod
    def create_simple_uschema() -> USchema:
        """Create a simple U-Schema for testing."""
        return USchema(
            entities=[
                USchemaEntity(
                    name="Customer",
                    attributes=[
                        USchemaAttribute("id", DataType.UUID, required=True),
                        USchemaAttribute("name", DataType.STRING, required=True),
                        USchemaAttribute("email", DataType.STRING, required=True),
                    ]
                )
            ],
            version="1.0"
        )
    
    @staticmethod
    def create_complex_uschema() -> USchema:
        """Create a complex U-Schema with relationships."""
        return USchema(
            entities=[
                USchemaEntity(
                    name="Customer",
                    attributes=[
                        USchemaAttribute("id", DataType.UUID, required=True),
                        USchemaAttribute("email", DataType.STRING, required=True),
                        USchemaAttribute("first_name", DataType.STRING, required=True),
                        USchemaAttribute("last_name", DataType.STRING, required=True),
                        USchemaAttribute("phone", DataType.STRING, required=False),
                        USchemaAttribute("created_at", DataType.TIMESTAMP, required=True),
                    ]
                ),
                USchemaEntity(
                    name="Order",
                    attributes=[
                        USchemaAttribute("id", DataType.UUID, required=True),
                        USchemaAttribute("customer_id", DataType.UUID, required=True),
                        USchemaAttribute("total", DataType.DECIMAL, required=True),
                        USchemaAttribute("status", DataType.STRING, required=True),
                        USchemaAttribute("order_date", DataType.TIMESTAMP, required=True),
                    ],
                    relationships=[
                        USchemaRelationship(
                            relationship_type="belongsTo",
                            target_entity="Customer",
                            foreign_key="customer_id"
                        )
                    ]
                ),
                USchemaEntity(
                    name="Product",
                    attributes=[
                        USchemaAttribute("id", DataType.UUID, required=True),
                        USchemaAttribute("sku", DataType.STRING, required=True),
                        USchemaAttribute("name", DataType.STRING, required=True),
                        USchemaAttribute("price", DataType.DECIMAL, required=True),
                        USchemaAttribute("stock_quantity", DataType.INTEGER, required=True),
                    ]
                )
            ],
            version="1.0"
        )
    
    @staticmethod
    def create_empty_schema() -> SchemaMetadata:
        """Create an empty database schema."""
        return SchemaMetadata(
            tables=[],
            database_name="test_db",
            version="15.2",
            introspection_timestamp=datetime.now()
        )
    
    @staticmethod
    def create_existing_schema() -> SchemaMetadata:
        """Create an existing database schema with tables."""
        return SchemaMetadata(
            tables=[
                Table(
                    name="customers",
                    schema="public",
                    columns=[
                        Column("id", "UUID", nullable=False, primary_key=True),
                        Column("email", "VARCHAR(255)", nullable=False, unique=True),
                        Column("first_name", "VARCHAR(100)", nullable=False),
                        Column("last_name", "VARCHAR(100)", nullable=False),
                        Column("created_at", "TIMESTAMP", nullable=False),
                    ],
                    primary_keys=["id"],
                    indexes=[
                        Index("idx_customers_email", ["email"], unique=True)
                    ],
                    row_count=1000
                ),
                Table(
                    name="products",
                    schema="public",
                    columns=[
                        Column("id", "UUID", nullable=False, primary_key=True),
                        Column("name", "VARCHAR(255)", nullable=False),
                        Column("price", "NUMERIC(10,2)", nullable=False),
                        Column("created_at", "TIMESTAMP", nullable=False),
                    ],
                    primary_keys=["id"],
                    row_count=500
                )
            ],
            database_name="test_db",
            version="15.2"
        )
    
    @staticmethod
    def create_schema_with_relationships() -> SchemaMetadata:
        """Create schema with foreign key relationships."""
        return SchemaMetadata(
            tables=[
                Table(
                    name="customers",
                    columns=[
                        Column("id", "UUID", nullable=False, primary_key=True),
                        Column("email", "VARCHAR(255)", nullable=False),
                    ],
                    primary_keys=["id"]
                ),
                Table(
                    name="orders",
                    columns=[
                        Column("id", "UUID", nullable=False, primary_key=True),
                        Column("customer_id", "UUID", nullable=False),
                        Column("total", "NUMERIC(10,2)", nullable=False),
                    ],
                    primary_keys=["id"],
                    foreign_keys=[
                        ForeignKey(
                            name="fk_orders_customer",
                            column="customer_id",
                            referenced_table="customers",
                            referenced_column="id"
                        )
                    ]
                )
            ],
            database_name="test_db"
        )
    
    @staticmethod
    def create_default_rules() -> RuleSet:
        """Create default rule set."""
        return RuleSet(
            naming=NamingConvention(
                table_pattern="snake_case",
                column_pattern="snake_case",
                pk_suffix="_id",
                fk_suffix="_id"
            ),
            design_rules=[
                DesignRule(
                    rule_id="R1_NAMING",
                    name="Naming Convention",
                    description="Tables must use snake_case",
                    category="naming",
                    severity="warning"
                ),
                DesignRule(
                    rule_id="R2_NO_NULL_FK",
                    name="Non-Nullable FKs",
                    description="Foreign keys should be NOT NULL",
                    category="structure",
                    severity="warning"
                )
            ],
            type_mappings={
                DataType.STRING: "VARCHAR(255)",
                DataType.INTEGER: "INTEGER",
                DataType.DECIMAL: "NUMERIC(10,2)",
                DataType.UUID: "UUID",
                DataType.TIMESTAMP: "TIMESTAMP"
            }
        )
    
    @staticmethod
    def create_sample_changes() -> List[SchemaChange]:
        """Create sample schema changes."""
        return [
            SchemaChange(
                change_type=ChangeType.CREATE_TABLE,
                target_table="orders",
                definition="id UUID, customer_id UUID, total NUMERIC(10,2)",
                reason="New entity 'Order' requires table",
                safe=True
            ),
            SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                target_table="customers",
                target_column="phone",
                definition="VARCHAR(20)",
                reason="New attribute 'phone' in U-Schema",
                safe=True
            ),
            SchemaChange(
                change_type=ChangeType.ADD_CONSTRAINT,
                target_table="orders",
                target_column="customer_id",
                definition="FOREIGN KEY (customer_id) REFERENCES customers(id)",
                reason="Relationship defined in U-Schema",
                safe=True
            )
        ]
    
    @staticmethod
    def load_json_fixture(filename: str) -> Dict[str, Any]:
        """Load JSON fixture from examples directory."""
        import os
        fixture_path = os.path.join("examples", filename)
        with open(fixture_path, 'r') as f:
            return json.load(f)
