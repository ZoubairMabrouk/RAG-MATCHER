"""
Knowledge Base Builder for MIMIC-III schema.
Builds the corpus of knowledge for RAG retrieval.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from src.domain.entities.rag_schema import (
    KnowledgeBaseDocument, 
    MIMICColumn, 
    MIMICSchemaMetadata
)

logger = logging.getLogger(__name__)


class MIMICKnowledgeBaseBuilder:
    """
    Builds knowledge base from MIMIC-III schema and dictionary.
    Single Responsibility: Knowledge base construction.
    """
    
    def __init__(self):
        self.medical_ontologies = self._load_medical_ontologies()
        self.technical_synonyms = self._load_technical_synonyms()
        
    def build_from_ddl(self, ddl_path: str, dictionary_path: str) -> List[KnowledgeBaseDocument]:
        """
        Build knowledge base from DDL and dictionary files.
        
        Args:
            ddl_path: Path to MIMIC-III DDL file
            dictionary_path: Path to data dictionary
            
        Returns:
            List of knowledge base documents
        """
        logger.info(f"Building knowledge base from DDL: {ddl_path}")
        
        # Parse DDL to extract schema structure
        schema_metadata = self._parse_ddl(ddl_path)
        
        # Load dictionary
        dictionary = self._load_dictionary(dictionary_path)
        
        # Generate documents for each column
        documents = []
        
        for table_name, columns in schema_metadata.columns.items():
            for column in columns:
                # Enhance column with dictionary information
                enhanced_column = self._enhance_with_dictionary(column, dictionary)
                
                # Create knowledge base document
                doc = self._create_document(enhanced_column)
                documents.append(doc)
                
                # Create additional documents for synonyms
                synonym_docs = self._create_synonym_documents(enhanced_column)
                documents.extend(synonym_docs)
        
        logger.info(f"Generated {len(documents)} knowledge base documents")
        return documents
    
    def _parse_ddl(self, ddl_path: str) -> MIMICSchemaMetadata:
        """Parse DDL file to extract schema structure."""
        with open(ddl_path, 'r', encoding='utf-8') as f:
            ddl_content = f.read()
        
        # Extract table and column information
        tables = []
        columns_by_table = {}
        relationships = {}
        
        # Simple DDL parsing (in production, use a proper SQL parser)
        table_pattern = r'CREATE TABLE\s+(\w+)\s*\('
        column_pattern = r'(\w+)\s+(\w+(?:\([^)]+\))?)\s*(.*?)(?:,|$)'
        
        current_table = None
        table_columns = []
        
        for line in ddl_content.split('\n'):
            line = line.strip()
            
            # Match table creation
            table_match = re.search(table_pattern, line, re.IGNORECASE)
            if table_match:
                if current_table:
                    columns_by_table[current_table] = table_columns
                    tables.append(current_table)
                current_table = table_match.group(1)
                table_columns = []
                continue
            
            # Match column definition
            if current_table and line:
                column_match = re.search(column_pattern, line)
                if column_match:
                    col_name = column_match.group(1)
                    col_type = column_match.group(2)
                    constraints = column_match.group(3).strip()
                    
                    # Parse constraints
                    constraint_dict = self._parse_constraints(constraints)
                    
                    column = MIMICColumn(
                        table=current_table,
                        column=col_name,
                        data_type=col_type,
                        description="",  # Will be filled from dictionary
                        constraints=constraint_dict
                    )
                    table_columns.append(column)
        
        # Add last table
        if current_table:
            columns_by_table[current_table] = table_columns
            tables.append(current_table)
        
        return MIMICSchemaMetadata(
            version="1.0",
            tables=tables,
            columns=columns_by_table,
            relationships=relationships
        )
    
    def _parse_constraints(self, constraints_str: str) -> Dict[str, Any]:
        """Parse column constraints."""
        constraints = {}
        
        if 'PRIMARY KEY' in constraints_str.upper():
            constraints['is_primary_key'] = True
        if 'FOREIGN KEY' in constraints_str.upper():
            constraints['is_foreign_key'] = True
        if 'NOT NULL' in constraints_str.upper():
            constraints['not_null'] = True
        if 'UNIQUE' in constraints_str.upper():
            constraints['unique'] = True
            
        return constraints
    
    def _load_dictionary(self, dictionary_path: str) -> Dict[str, Dict[str, Any]]:
        """Load data dictionary."""
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Dictionary file not found: {dictionary_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing dictionary: {e}")
            return {}
    
    def _enhance_with_dictionary(self, column: MIMICColumn, dictionary: Dict[str, Dict[str, Any]]) -> MIMICColumn:
        """Enhance column with dictionary information."""
        table_col_key = f"{column.table}.{column.column}"
        
        if table_col_key in dictionary:
            dict_info = dictionary[table_col_key]
            
            # Update description
            if 'description' in dict_info:
                column.description = dict_info['description']
            
            # Add synonyms
            if 'synonyms' in dict_info:
                column.synonyms.extend(dict_info['synonyms'])
            
            # Add units
            if 'units' in dict_info:
                column.units = dict_info['units']
            
            # Add value profile
            if 'value_profile' in dict_info:
                column.value_profile.update(dict_info['value_profile'])
            
            # Add examples
            if 'examples' in dict_info:
                column.examples_synth.extend(dict_info['examples'])
        
        # Add medical ontologies
        column.synonyms.extend(self._get_medical_synonyms(column.column))
        
        # Add technical synonyms
        column.synonyms.extend(self._get_technical_synonyms(column.column))
        
        return column
    
    def _create_document(self, column: MIMICColumn) -> KnowledgeBaseDocument:
        """Create a knowledge base document for a column."""
        content_parts = [
            f"Table: {column.table}",
            f"Column: {column.column}",
            f"Type: {column.data_type}",
            f"Description: {column.description}",
        ]
        
        if column.synonyms:
            content_parts.append(f"Synonyms: {', '.join(column.synonyms)}")
        
        if column.units:
            content_parts.append(f"Units: {column.units}")
        
        if column.constraints:
            constraints_str = ', '.join([f"{k}: {v}" for k, v in column.constraints.items()])
            content_parts.append(f"Constraints: {constraints_str}")
        
        if column.value_profile:
            profile_str = ', '.join([f"{k}: {v}" for k, v in column.value_profile.items()])
            content_parts.append(f"Value Profile: {profile_str}")
        
        content = ". ".join(content_parts)
        
        return KnowledgeBaseDocument(
            id=f"{column.table}.{column.column}",
            table=column.table,
            column=column.column,
            content=content,
            metadata={
                "data_type": column.data_type,
                "description": column.description,
                "synonyms": column.synonyms,
                "units": column.units,
                "constraints": column.constraints,
                "value_profile": column.value_profile,
                "examples": column.examples_synth
            }
        )
    
    def _create_synonym_documents(self, column: MIMICColumn) -> List[KnowledgeBaseDocument]:
        """Create additional documents for synonyms."""
        synonym_docs = []
        
        for synonym in column.synonyms[:3]:  # Limit to top 3 synonyms
            content = f"Synonym for {column.table}.{column.column}: {synonym}. {column.description}"
            
            doc = KnowledgeBaseDocument(
                id=f"{column.table}.{column.column}.synonym.{synonym}",
                table=column.table,
                column=column.column,
                content=content,
                metadata={
                    "synonym_for": f"{column.table}.{column.column}",
                    "synonym": synonym,
                    "data_type": column.data_type,
                    "description": column.description
                }
            )
            synonym_docs.append(doc)
        
        return synonym_docs
    
    def _load_medical_ontologies(self) -> Dict[str, List[str]]:
        """Load medical ontologies and synonyms."""
        return {
            # Vital signs
            "heart_rate": ["hr", "heart rate", "pulse", "bpm", "beats per minute"],
            "blood_pressure": ["bp", "blood pressure", "systolic", "diastolic", "mmHg"],
            "temperature": ["temp", "temperature", "fever", "°C", "°F"],
            "respiratory_rate": ["rr", "respiratory rate", "breathing rate", "resp/min"],
            "oxygen_saturation": ["spo2", "oxygen sat", "oxygen saturation", "%"],
            
            # Patient identifiers
            "patient_id": ["patient", "subject", "person", "individual", "pt_id"],
            "admission_id": ["admission", "encounter", "visit", "hadm_id", "enc_id"],
            "icu_stay_id": ["icu", "intensive care", "stay", "icustay_id"],
            
            # Dates and times
            "admission_time": ["adm_time", "admit_time", "admission", "admitted"],
            "discharge_time": ["disch_time", "discharge", "discharged"],
            "birth_date": ["dob", "date of birth", "birth", "anchor_age"],
            "death_time": ["dod", "date of death", "death", "deceased"],
            
            # Clinical codes
            "icd9_code": ["icd9", "diagnosis code", "dx_code", "diag_code"],
            "icd10_code": ["icd10", "diagnosis code", "dx_code", "diag_code"],
            "drg_code": ["drg", "diagnosis related group"],
            "item_id": ["item", "measurement", "observation", "chart_item"],
            
            # Medications
            "drug": ["medication", "med", "drug_name", "prescription"],
            "ndc": ["national drug code", "drug_code", "medication_code"],
            
            # Lab values
            "lab_value": ["laboratory", "lab", "test", "result", "value"],
            "lab_units": ["unit", "measurement unit", "uom", "valueuom"],
        }
    
    def _load_technical_synonyms(self) -> Dict[str, List[str]]:
        """Load technical synonyms."""
        return {
            "id": ["identifier", "key", "pk", "primary key"],
            "name": ["label", "title", "description"],
            "date": ["timestamp", "datetime", "time"],
            "count": ["number", "quantity", "amount"],
            "status": ["state", "condition", "flag"],
            "type": ["category", "class", "kind"],
            "code": ["identifier", "reference", "value"],
        }
    
    def _get_medical_synonyms(self, column_name: str) -> List[str]:
        """Get medical synonyms for a column name."""
        synonyms = []
        column_lower = column_name.lower()
        
        for key, values in self.medical_ontologies.items():
            if any(term in column_lower for term in [key] + values):
                synonyms.extend(values)
        
        return list(set(synonyms))[:5]  # Limit and deduplicate
    
    def _get_technical_synonyms(self, column_name: str) -> List[str]:
        """Get technical synonyms for a column name."""
        synonyms = []
        column_lower = column_name.lower()
        
        for key, values in self.technical_synonyms.items():
            if key in column_lower:
                synonyms.extend(values)
        
        return list(set(synonyms))[:3]  # Limit and deduplicate
    
    def save_documents(self, documents: List[KnowledgeBaseDocument], output_path: str) -> None:
        """Save documents to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc.dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(documents)} documents to {output_path}")
    
    def load_documents(self, input_path: str) -> List[KnowledgeBaseDocument]:
        """Load documents from JSONL file."""
        documents = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line)
                    documents.append(KnowledgeBaseDocument(**doc_data))
        
        logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents

