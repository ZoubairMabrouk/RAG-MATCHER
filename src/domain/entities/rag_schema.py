"""
RAG-specific domain entities for schema matching.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class FieldType(str, Enum):
    """Supported field types for matching."""
    DATETIME = "datetime"
    INTEGER = "integer"
    FLOAT = "float"
    TEXT = "text"
    BOOLEAN = "boolean"
    CODE = "code"  # ICD codes, item IDs, etc.
    ID = "id"  # Patient IDs, admission IDs, etc.


class DecisionAction(str, Enum):
    """Possible decision actions."""
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class SourceField(BaseModel):
    """Represents a source NoSQL field for matching."""
    path: str = Field(..., description="Full path to the field (e.g., 'encounters[0].adm_date')")
    name_tokens: List[str] = Field(..., description="Tokenized field name")
    inferred_type: FieldType = Field(..., description="Inferred data type")
    format_regex: Optional[str] = Field(None, description="Detected format pattern")
    units: Optional[str] = Field(None, description="Detected units (mmHg, bpm, etc.)")
    category_values: Optional[List[str]] = Field(None, description="Sample categorical values")
    hints: List[str] = Field(default_factory=list, description="Semantic hints")
    neighbors: List[str] = Field(default_factory=list, description="Nearby field names")
    coarse_semantics: List[str] = Field(default_factory=list, description="High-level semantic categories")
    
    class Config:
        use_enum_values = True


class MIMICColumn(BaseModel):
    """Represents a MIMIC-III column with metadata."""
    table: str = Field(..., description="Table name")
    column: str = Field(..., description="Column name")
    data_type: str = Field(..., description="SQL data type")
    description: str = Field(..., description="Column description from dictionary")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints (PK, FK, NOT NULL, etc.)")
    fk_context: Dict[str, str] = Field(default_factory=dict, description="Foreign key relationships")
    examples_synth: List[Union[str, int, float]] = Field(default_factory=list, description="Synthetic examples")
    synonyms: List[str] = Field(default_factory=list, description="Synonyms and abbreviations")
    units: Optional[str] = Field(None, description="Expected units")
    value_profile: Dict[str, Any] = Field(default_factory=dict, description="Statistical profile")
    neighbors: List[str] = Field(default_factory=list, description="Columns often used together")
    join_paths: List[Dict[str, str]] = Field(default_factory=list, description="Common join patterns")


class CandidateMatch(BaseModel):
    """Represents a candidate match between source and target."""
    target: str = Field(..., description="Target table.column")
    confidence_model: float = Field(..., description="Model confidence (0-1)")
    confidence_llm: float = Field(..., description="LLM confidence (0-1)")
    rationale: str = Field(..., description="Explanation for the match")
    guardrails: List[str] = Field(default_factory=list, description="Applied validation rules")


class MatchingDecision(BaseModel):
    """Final matching decision."""
    action: DecisionAction = Field(..., description="Decision action")
    selected_target: Optional[str] = Field(None, description="Selected target column")
    final_confidence: float = Field(..., description="Final confidence score")
    guardrails: List[str] = Field(default_factory=list, description="Applied guardrails")
    review_checklist: Optional[List[str]] = Field(None, description="Items for human review")


class SchemaMatchingResult(BaseModel):
    """Complete result of schema matching for a source field."""
    source_field: str = Field(..., description="Source field path")
    candidates: List[CandidateMatch] = Field(..., description="Candidate matches")
    decision: MatchingDecision = Field(..., description="Final decision")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Version of models used")
    
    class Config:
        use_enum_values = True


class KnowledgeBaseDocument(BaseModel):
    """Document in the RAG knowledge base."""
    id: str = Field(..., description="Unique document ID")
    table: str = Field(..., description="Table name")
    column: str = Field(..., description="Column name")
    content: str = Field(..., description="Searchable content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_text(self) -> str:
        """Convert to searchable text format."""
        return f"{self.table}.{self.column}: {self.content}"


class RetrievalQuery(BaseModel):
    """Query object for retrieval."""
    source_field: SourceField = Field(..., description="Source field to match")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Retrieval filters")
    top_k: int = Field(20, description="Number of candidates to retrieve")


class ScoringWeights(BaseModel):
    """Weights for hybrid scoring."""
    bi_encoder: float = Field(0.45, description="Bi-encoder similarity weight")
    cross_encoder: float = Field(0.35, description="Cross-encoder reranking weight")
    llm_confidence: float = Field(0.15, description="LLM confidence weight")
    constraints: float = Field(0.05, description="Constraints validation weight")


class ScoringThresholds(BaseModel):
    """Thresholds for decision making."""
    accept_high: float = Field(0.78, description="High threshold for ACCEPT")
    review_low: float = Field(0.55, description="Low threshold for REVIEW")
    reject_below: float = Field(0.55, description="Below this = REJECT")


class MIMICSchemaMetadata(BaseModel):
    """Complete MIMIC-III schema metadata."""
    version: str = Field(..., description="Schema version")
    tables: List[str] = Field(..., description="List of table names")
    columns: Dict[str, List[MIMICColumn]] = Field(..., description="Columns by table")
    relationships: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict, description="Table relationships")
    ontologies: Dict[str, List[str]] = Field(default_factory=dict, description="Medical ontologies and synonyms")
    value_domains: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Value domain constraints")

