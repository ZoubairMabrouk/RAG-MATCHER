from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime


class DataType(Enum):
    """Standard data types across different database systems."""
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    DATE = "date"
    JSON = "json"
    UUID = "uuid"


class ChangeType(Enum):
    """Types of schema changes."""
    CREATE_TABLE = "create_table"
    #DROP_TABLE = "drop_table"
    ADD_COLUMN = "add_column"
    #DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    ADD_INDEX = "add_index"
    #DROP_INDEX = "drop_index"
    ADD_CONSTRAINT = "add_constraint"
    #DROP_CONSTRAINT = "drop_constraint"
    CREATE_VIEW = "create_view"
    #DROP_VIEW = "drop_view"
    # STEP 7 additions (EvolutionPlanner): kept distinct from ADD_CONSTRAINT so
    # evolution reports can tell "generic constraint" apart from "FK introduced
    # by an EVOLVE decision" and "new association/junction table introduced by
    # an EVOLVE decision". MigrationBuilder still lowers these to SQL DDL.
    ADD_FOREIGN_KEY = "add_foreign_key"
    ADD_ASSOCIATION_TABLE = "add_association_table"


@dataclass(frozen=True)
class USchemaAttribute:
    """Represents an attribute in U-Schema.

    STEP 5.2 extension (see docs/experimental_architecture.md, section 3):
    added structural fields required by MDEValidator to distinguish
    object/array/scalar and nesting, without breaking DataType or any
    existing positional instantiation (`USchemaAttribute(name, data_type, required)`
    still works everywhere in the repo since new fields all have defaults
    and are appended after the existing ones).
    """
    name: str
    data_type: DataType
    required: bool = False
    description: Optional[str] = None
    is_key: bool = False
    constraints: Dict[str, Any] = field(default_factory=dict)
    # --- structural metadata (new, additive) ---
    is_object: bool = False
    """True if this attribute represents a nested object (not a leaf scalar)."""
    is_array: bool = False
    """True if this attribute represents an array/list of values or objects."""
    nesting_path: Tuple[str, ...] = field(default_factory=tuple)
    """Full dotted path from the entity root, e.g. ('patient', 'device', 'id')."""
    parent_path: Optional[str] = None
    """Dotted path of the immediate parent object, e.g. 'patient.device'. None if top-level."""
    structural_role: Optional[str] = None
    """Optional free-form role hint: 'identifier', 'foreign_reference', 'measure', etc."""

    @property
    def is_scalar(self) -> bool:
        """True if this attribute is a leaf scalar value (not object/array)."""
        return not self.is_object and not self.is_array

    @property
    def depth(self) -> int:
        """Nesting depth (0 = top-level attribute)."""
        return max(len(self.nesting_path) - 1, 0)


@dataclass(frozen=True)
class USchemaRelationship:
    """Represents a relationship in U-Schema."""
    relationship_type: str  # belongsTo, hasMany, manyToMany
    target_entity: str
    foreign_key: Optional[str] = None
    inverse_key: Optional[str] = None


@dataclass(frozen=True)
class USchemaEntity:
    """Represents an entity in U-Schema (NoSQL-oriented conceptual model)."""
    name: str
    attributes: List[USchemaAttribute]
    relationships: List[USchemaRelationship] = field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class USchema:
    """Root U-Schema model."""
    entities: List[USchemaEntity]
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Column:
    """Represents a database column."""
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    default_value: Optional[Any] = None
    comment: Optional[str] = None
    constraints: List[str] = field(default_factory=list)


@dataclass
class ForeignKey:
    """Represents a foreign key constraint."""
    name: str
    column: str
    referenced_table: str
    referenced_column: str
    on_delete: str = "NO ACTION"
    on_update: str = "NO ACTION"


@dataclass
class Index:
    """Represents a database index."""
    name: str
    columns: List[str]
    unique: bool = False
    index_type: Optional[str] = None  # BTREE, HASH, etc.


@dataclass
class Table:
    """Represents a database table."""
    name: str
    schema: str = "public"
    columns: List[Column] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    indexes: List[Index] = field(default_factory=list)
    comment: Optional[str] = None
    row_count: int = 0
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None


@dataclass
class SchemaMetadata:
    """Complete database schema with metadata."""
    tables: List[Table]
    views: List[Dict[str, Any]] = field(default_factory=list)
    materialized_views: List[Dict[str, Any]] = field(default_factory=list)
    database_name: str = ""
    version: str = ""
    introspection_timestamp: Optional[datetime] = None