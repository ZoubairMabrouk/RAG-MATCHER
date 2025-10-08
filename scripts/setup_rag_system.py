"""
Setup script for RAG system initialization.
Builds knowledge base, initializes models, and creates demo data.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.infrastructure.rag.knowledge_base_builder import MIMICKnowledgeBaseBuilder
from src.infrastructure.rag.embedding_service import RAGEmbeddingService
from src.infrastructure.rag.vector_store import RAGVectorStore
from src.infrastructure.rag.retriever import AdvancedRAGRetriever
from src.infrastructure.rag.llm_orchestrator import LLMOrchestrator
from src.infrastructure.rag.scoring_system import HybridScoringSystem
from src.infrastructure.rag.rag_orchestrator import RAGOrchestrator, RAGService
from src.infrastructure.rag.evaluation_metrics import EvaluationDataset
from src.infrastructure.llm.llm_client import ILLMClient
from src.domain.entities.rag_schema import (
    SourceField, FieldType, ScoringWeights, ScoringThresholds
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockLLMClient(ILLMClient):
    """Mock LLM client for demonstration purposes."""
    
    def generate_response(self, system_prompt: str, user_prompt: str, 
                         temperature: float = 0.0, max_tokens: int = 1024,
                         response_format: Optional[Dict] = None) -> str:
        """Generate mock response."""
        # Mock response for demonstration
        return json.dumps({
            "source_field": "patient.id",
            "candidates": [
                {
                    "target": "PATIENTS.SUBJECT_ID",
                    "confidence_model": 0.85,
                    "confidence_llm": 0.82,
                    "rationale": "Patient identifier matches SUBJECT_ID in PATIENTS table"
                }
            ],
            "decision": {
                "action": "ACCEPT",
                "selected_target": "PATIENTS.SUBJECT_ID",
                "final_confidence": 0.84,
                "guardrails": ["type:id", "table:PATIENTS"]
            }
        })


def create_sample_mimic_ddl() -> str:
    """Create sample MIMIC-III DDL for demonstration."""
    ddl_content = """
-- Sample MIMIC-III DDL for demonstration
CREATE TABLE PATIENTS (
    SUBJECT_ID INTEGER PRIMARY KEY,
    GENDER VARCHAR(5),
    DOB DATE,
    DOD DATE,
    DOD_HOSP DATE,
    DOD_SSN DATE,
    EXPIRE_FLAG INTEGER
);

CREATE TABLE ADMISSIONS (
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER PRIMARY KEY,
    ADMITTIME TIMESTAMP,
    DISCHTIME TIMESTAMP,
    DEATHTIME TIMESTAMP,
    ADMISSION_TYPE VARCHAR(50),
    ADMISSION_LOCATION VARCHAR(50),
    DISCHARGE_LOCATION VARCHAR(50),
    INSURANCE VARCHAR(255),
    LANGUAGE VARCHAR(10),
    RELIGION VARCHAR(50),
    MARITAL_STATUS VARCHAR(50),
    ETHNICITY VARCHAR(200),
    EDREGTIME TIMESTAMP,
    EDOUTTIME TIMESTAMP,
    DIAGNOSIS TEXT,
    HOSPITAL_EXPIRE_FLAG INTEGER,
    HAS_CHARTEVENTS_DATA INTEGER,
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID)
);

CREATE TABLE ICUSTAYS (
    ROW_ID INTEGER PRIMARY KEY,
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER,
    ICUSTAY_ID INTEGER,
    DBSOURCE VARCHAR(20),
    FIRST_CAREUNIT VARCHAR(20),
    LAST_CAREUNIT VARCHAR(20),
    FIRST_WARDID INTEGER,
    LAST_WARDID INTEGER,
    INTIME TIMESTAMP,
    OUTTIME TIMESTAMP,
    LOS DECIMAL,
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID),
    FOREIGN KEY (HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

CREATE TABLE CHARTEVENTS (
    ROW_ID INTEGER PRIMARY KEY,
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER,
    ICUSTAY_ID INTEGER,
    ITEMID INTEGER,
    CHARTTIME TIMESTAMP,
    STORETIME TIMESTAMP,
    CGID INTEGER,
    VALUE TEXT,
    VALUENUM DECIMAL,
    VALUEUOM VARCHAR(30),
    WARNING INTEGER,
    ERROR INTEGER,
    RESULTSTATUS VARCHAR(50),
    STOPPED VARCHAR(30),
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID),
    FOREIGN KEY (HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY (ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID)
);

CREATE TABLE LABEVENTS (
    ROW_ID INTEGER PRIMARY KEY,
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER,
    ITEMID INTEGER,
    CHARTTIME TIMESTAMP,
    VALUE TEXT,
    VALUENUM DECIMAL,
    VALUEUOM VARCHAR(30),
    FLAG VARCHAR(30),
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID),
    FOREIGN KEY (HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

CREATE TABLE DIAGNOSES_ICD (
    ROW_ID INTEGER PRIMARY KEY,
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER,
    SEQ_NUM INTEGER,
    ICD9_CODE VARCHAR(10),
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID),
    FOREIGN KEY (HADM_ID) REFERENCES ADMISSIONS(HADM_ID)
);

CREATE TABLE PRESCRIPTIONS (
    ROW_ID INTEGER PRIMARY KEY,
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER,
    ICUSTAY_ID INTEGER,
    STARTDATE TIMESTAMP,
    ENDDATE TIMESTAMP,
    DRUG_TYPE VARCHAR(100),
    DRUG VARCHAR(100),
    DRUG_NAME_POE VARCHAR(100),
    DRUG_NAME_GENERIC VARCHAR(100),
    FORMULARY_DRUG_CD VARCHAR(120),
    GSN VARCHAR(200),
    NDC VARCHAR(120),
    PROD_STRENGTH VARCHAR(120),
    DOSE_VAL_RX VARCHAR(120),
    DOSE_UNIT_RX VARCHAR(120),
    FORM_VAL_DISP VARCHAR(120),
    FORM_UNIT_DISP VARCHAR(120),
    ROUTE VARCHAR(120),
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID),
    FOREIGN KEY (HADM_ID) REFERENCES ADMISSIONS(HADM_ID),
    FOREIGN KEY (ICUSTAY_ID) REFERENCES ICUSTAYS(ICUSTAY_ID)
);
"""
    return ddl_content


def create_sample_dictionary() -> Dict[str, Any]:
    """Create sample data dictionary for demonstration."""
    return {
        "PATIENTS.SUBJECT_ID": {
            "description": "Unique identifier for each patient",
            "synonyms": ["patient_id", "subject", "person_id"],
            "units": None,
            "value_profile": {"type": "integer", "unique": True}
        },
        "PATIENTS.GENDER": {
            "description": "Patient gender (M/F)",
            "synonyms": ["sex", "gender_code"],
            "units": None,
            "value_profile": {"type": "categorical", "values": ["M", "F"]}
        },
        "ADMISSIONS.HADM_ID": {
            "description": "Unique identifier for each admission",
            "synonyms": ["admission_id", "visit_id", "encounter_id"],
            "units": None,
            "value_profile": {"type": "integer", "unique": True}
        },
        "ADMISSIONS.ADMITTIME": {
            "description": "Date and time of hospital admission",
            "synonyms": ["admission_time", "admit_time", "start_time"],
            "units": "timestamp",
            "value_profile": {"type": "datetime", "format": "YYYY-MM-DD HH:MM:SS"}
        },
        "CHARTEVENTS.ITEMID": {
            "description": "Identifier for the measurement type",
            "synonyms": ["measurement_id", "chart_item", "observation_id"],
            "units": None,
            "value_profile": {"type": "integer", "coded": True}
        },
        "CHARTEVENTS.VALUENUM": {
            "description": "Numeric value of the measurement",
            "synonyms": ["numeric_value", "value", "measurement_value"],
            "units": "varies",
            "value_profile": {"type": "numeric", "nullable": True}
        },
        "CHARTEVENTS.VALUEUOM": {
            "description": "Unit of measurement",
            "synonyms": ["unit", "measurement_unit", "uom"],
            "units": None,
            "value_profile": {"type": "text", "categorical": True}
        },
        "LABEVENTS.VALUENUM": {
            "description": "Numeric value of the laboratory test",
            "synonyms": ["lab_value", "test_result", "numeric_result"],
            "units": "varies",
            "value_profile": {"type": "numeric", "nullable": True}
        },
        "DIAGNOSES_ICD.ICD9_CODE": {
            "description": "ICD-9 diagnosis code",
            "synonyms": ["diagnosis_code", "icd_code", "dx_code"],
            "units": None,
            "value_profile": {"type": "code", "format": "ICD9"}
        },
        "PRESCRIPTIONS.DRUG": {
            "description": "Name of the prescribed drug",
            "synonyms": ["medication", "drug_name", "prescription"],
            "units": None,
            "value_profile": {"type": "text", "free_text": True}
        }
    }


def setup_rag_system(data_dir: str = "data/rag") -> RAGService:
    """Setup complete RAG system."""
    logger.info("Setting up RAG system...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Create sample MIMIC-III DDL and dictionary
    logger.info("Creating sample MIMIC-III schema...")
    ddl_path = os.path.join(data_dir, "mimic_ddl.sql")
    dict_path = os.path.join(data_dir, "mimic_dictionary.json")
    
    with open(ddl_path, 'w') as f:
        f.write(create_sample_mimic_ddl())
    
    with open(dict_path, 'w') as f:
        json.dump(create_sample_dictionary(), f, indent=2)
    
    # Step 2: Build knowledge base
    logger.info("Building knowledge base...")
    kb_builder = MIMICKnowledgeBaseBuilder()
    documents = kb_builder.build_from_ddl(ddl_path, dict_path)
    
    # Save documents
    docs_path = os.path.join(data_dir, "knowledge_base.jsonl")
    kb_builder.save_documents(documents, docs_path)
    
    # Step 3: Initialize embedding service
    logger.info("Initializing embedding service...")
    embedding_service = RAGEmbeddingService()
    
    # Step 4: Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_service.embed_documents(documents)
    
    # Step 5: Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = RAGVectorStore(
        dimension=embedding_service.get_embedding_dimension(),
        index_type="Flat"  # Use Flat for demo
    )
    vector_store.add_documents(documents, embeddings)
    
    # Save vector store
    vector_store.save(os.path.join(data_dir, "vector_store"))
    
    # Step 6: Initialize LLM client (mock for demo)
    logger.info("Initializing LLM client...")
    llm_client = MockLLMClient()
    
    # Step 7: Initialize scoring configuration
    weights = ScoringWeights(
        bi_encoder=0.45,
        cross_encoder=0.35,
        llm_confidence=0.15,
        constraints=0.05
    )
    
    thresholds = ScoringThresholds(
        accept_high=0.78,
        review_low=0.55,
        reject_below=0.55
    )
    
    # Step 8: Initialize orchestrator
    logger.info("Initializing RAG orchestrator...")
    orchestrator = RAGOrchestrator(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=llm_client,
        weights=weights,
        thresholds=thresholds
    )
    
    # Step 9: Create RAG service
    rag_service = RAGService(orchestrator)
    
    logger.info("RAG system setup completed!")
    return rag_service


def create_demo_dataset(data_dir: str = "data/rag") -> None:
    """Create demonstration dataset."""
    logger.info("Creating demo dataset...")
    
    dataset = EvaluationDataset()
    dataset.create_synthetic_dataset(num_cases=50)
    
    # Save dataset
    dataset_path = os.path.join(data_dir, "demo_dataset.json")
    dataset.save_dataset(dataset_path)
    
    # Print statistics
    stats = dataset.get_statistics()
    logger.info(f"Demo dataset created: {stats}")


def run_demo_matching(rag_service: RAGService) -> None:
    """Run demonstration matching."""
    logger.info("Running demo matching...")
    
    # Create sample source fields
    source_fields = [
        SourceField(
            path="patient.id",
            name_tokens=["patient", "id"],
            inferred_type=FieldType.ID,
            hints=["patient identifier"],
            coarse_semantics=["identifier"]
        ),
        SourceField(
            path="admission.date",
            name_tokens=["admission", "date"],
            inferred_type=FieldType.DATETIME,
            hints=["hospital admission", "visit start"],
            coarse_semantics=["temporal", "event_start"]
        ),
        SourceField(
            path="vitals.heart_rate",
            name_tokens=["heart", "rate"],
            inferred_type=FieldType.INTEGER,
            units="bpm",
            hints=["vital signs", "cardiac"],
            coarse_semantics=["measurement", "vital"]
        ),
        SourceField(
            path="labs.hemoglobin",
            name_tokens=["hemoglobin", "hgb"],
            inferred_type=FieldType.FLOAT,
            units="g/dL",
            hints=["laboratory", "blood test"],
            coarse_semantics=["measurement", "laboratory"]
        ),
        SourceField(
            path="diagnosis.code",
            name_tokens=["diagnosis", "code"],
            inferred_type=FieldType.CODE,
            hints=["medical diagnosis", "ICD"],
            coarse_semantics=["code", "diagnosis"]
        )
    ]
    
    # Perform matching
    results = rag_service._orchestrator.match_schema_fields(source_fields)
    
    # Display results
    for result in results:
        print(f"\nField: {result.source_field}")
        print(f"Decision: {result.decision.action}")
        print(f"Confidence: {result.decision.final_confidence:.3f}")
        if result.decision.selected_target:
            print(f"Target: {result.decision.selected_target}")
        if result.candidates:
            print("Top candidates:")
            for i, candidate in enumerate(result.candidates[:3]):
                print(f"  {i+1}. {candidate.target} (score: {candidate.confidence_model:.3f})")
                print(f"     {candidate.rationale}")


def main():
    """Main setup function."""
    try:
        # Setup RAG system
        rag_service = setup_rag_system()
        
        # Create demo dataset
        create_demo_dataset()
        
        # Run demo
        run_demo_matching(rag_service)
        
        # Get statistics
        stats = rag_service.get_matching_statistics()
        print(f"\nSystem Statistics:")
        print(json.dumps(stats, indent=2))
        
        logger.info("Setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()

