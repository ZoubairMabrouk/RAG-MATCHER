"""
Example usage of RAG schema matching system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.domain.entities.rag_schema import SourceField, FieldType
from src.infrastructure.rag.rag_orchestrator import RAGService


def create_sample_source_fields():
    """Create sample source fields for demonstration."""
    return [
        # Patient identifiers
        SourceField(
            path="patient.id",
            name_tokens=["patient", "id"],
            inferred_type=FieldType.ID,
            hints=["patient identifier", "primary key"],
            coarse_semantics=["identifier", "primary_key"]
        ),
        
        # Admission information
        SourceField(
            path="admission.date",
            name_tokens=["admission", "date"],
            inferred_type=FieldType.DATETIME,
            format_regex="YYYY-MM-DD",
            hints=["hospital admission", "visit start"],
            coarse_semantics=["temporal", "event_start"]
        ),
        
        SourceField(
            path="admission.id",
            name_tokens=["admission", "id"],
            inferred_type=FieldType.ID,
            hints=["admission identifier", "visit id"],
            coarse_semantics=["identifier", "foreign_key"]
        ),
        
        # Vital signs
        SourceField(
            path="vitals.heart_rate",
            name_tokens=["heart", "rate"],
            inferred_type=FieldType.INTEGER,
            units="bpm",
            hints=["vital signs", "cardiac", "pulse"],
            coarse_semantics=["measurement", "vital", "cardiac"]
        ),
        
        SourceField(
            path="vitals.blood_pressure_systolic",
            name_tokens=["blood", "pressure", "systolic"],
            inferred_type=FieldType.INTEGER,
            units="mmHg",
            hints=["vital signs", "cardiovascular"],
            coarse_semantics=["measurement", "vital", "cardiovascular"]
        ),
        
        SourceField(
            path="vitals.temperature",
            name_tokens=["temperature", "temp"],
            inferred_type=FieldType.FLOAT,
            units="°C",
            hints=["vital signs", "fever"],
            coarse_semantics=["measurement", "vital", "thermal"]
        ),
        
        # Laboratory values
        SourceField(
            path="labs.hemoglobin",
            name_tokens=["hemoglobin", "hgb"],
            inferred_type=FieldType.FLOAT,
            units="g/dL",
            hints=["laboratory", "blood test", "anemia"],
            coarse_semantics=["measurement", "laboratory", "hematology"]
        ),
        
        SourceField(
            path="labs.creatinine",
            name_tokens=["creatinine"],
            inferred_type=FieldType.FLOAT,
            units="mg/dL",
            hints=["laboratory", "kidney function"],
            coarse_semantics=["measurement", "laboratory", "renal"]
        ),
        
        SourceField(
            path="labs.units",
            name_tokens=["units", "unit"],
            inferred_type=FieldType.TEXT,
            hints=["measurement unit"],
            coarse_semantics=["metadata", "unit"]
        ),
        
        # Diagnoses
        SourceField(
            path="diagnosis.code",
            name_tokens=["diagnosis", "code"],
            inferred_type=FieldType.CODE,
            hints=["medical diagnosis", "ICD", "diagnostic code"],
            coarse_semantics=["code", "diagnosis", "medical"]
        ),
        
        SourceField(
            path="diagnosis.description",
            name_tokens=["diagnosis", "description"],
            inferred_type=FieldType.TEXT,
            hints=["diagnosis description", "diagnostic text"],
            coarse_semantics=["text", "diagnosis", "description"]
        ),
        
        # Medications
        SourceField(
            path="medication.drug_name",
            name_tokens=["drug", "name", "medication"],
            inferred_type=FieldType.TEXT,
            hints=["prescription", "medication", "drug"],
            coarse_semantics=["text", "medication", "prescription"]
        ),
        
        SourceField(
            path="medication.dosage",
            name_tokens=["dosage", "dose"],
            inferred_type=FieldType.FLOAT,
            units="mg",
            hints=["medication dosage", "drug dose"],
            coarse_semantics=["measurement", "medication", "dose"]
        ),
        
        # ICU information
        SourceField(
            path="icu.stay_id",
            name_tokens=["icu", "stay", "id"],
            inferred_type=FieldType.ID,
            hints=["intensive care", "ICU stay"],
            coarse_semantics=["identifier", "icu", "stay"]
        ),
        
        SourceField(
            path="icu.admission_time",
            name_tokens=["icu", "admission", "time"],
            inferred_type=FieldType.DATETIME,
            hints=["ICU admission", "intensive care start"],
            coarse_semantics=["temporal", "icu", "admission"]
        )
    ]


def demonstrate_single_field_matching(rag_service: RAGService):
    """Demonstrate single field matching."""
    print("=== Single Field Matching Demo ===")
    
    # Create a sample field
    source_field = SourceField(
        path="patient.heart_rate",
        name_tokens=["patient", "heart", "rate"],
        inferred_type=FieldType.INTEGER,
        units="bpm",
        hints=["vital signs", "cardiac"],
        coarse_semantics=["measurement", "vital"]
    )
    
    # Perform matching
    result = rag_service._orchestrator.match_single_field(source_field)
    
    # Display result
    print(f"Source Field: {result.source_field}")
    print(f"Decision: {result.decision.action}")
    print(f"Confidence: {result.decision.final_confidence:.3f}")
    
    if result.decision.selected_target:
        print(f"Selected Target: {result.decision.selected_target}")
    
    if result.candidates:
        print("\nTop Candidates:")
        for i, candidate in enumerate(result.candidates[:5]):
            print(f"  {i+1}. {candidate.target}")
            print(f"     Confidence: {candidate.confidence_model:.3f}")
            print(f"     Rationale: {candidate.rationale}")
            print()
    
    if result.decision.guardrails:
        print(f"Guardrails: {', '.join(result.decision.guardrails)}")
    
    if result.decision.review_checklist:
        print("\nReview Checklist:")
        for item in result.decision.review_checklist:
            print(f"  - {item}")


def demonstrate_batch_matching(rag_service: RAGService):
    """Demonstrate batch field matching."""
    print("\n=== Batch Field Matching Demo ===")
    
    # Create sample fields
    source_fields = create_sample_source_fields()
    
    # Perform batch matching
    results = rag_service._orchestrator.match_schema_fields(source_fields)
    
    # Display summary statistics
    total_fields = len(results)
    accept_count = sum(1 for r in results if r.decision.action.value == "ACCEPT")
    review_count = sum(1 for r in results if r.decision.action.value == "REVIEW")
    reject_count = sum(1 for r in results if r.decision.action.value == "REJECT")
    
    print(f"Total Fields Processed: {total_fields}")
    print(f"Accept: {accept_count} ({accept_count/total_fields*100:.1f}%)")
    print(f"Review: {review_count} ({review_count/total_fields*100:.1f}%)")
    print(f"Reject: {reject_count} ({reject_count/total_fields*100:.1f}%)")
    
    # Display detailed results
    print("\nDetailed Results:")
    for result in results:
        print(f"\n{result.source_field}:")
        print(f"  Decision: {result.decision.action.value}")
        print(f"  Confidence: {result.decision.final_confidence:.3f}")
        
        if result.decision.selected_target:
            print(f"  Target: {result.decision.selected_target}")
        
        if result.candidates:
            best_candidate = result.candidates[0]
            print(f"  Best Match: {best_candidate.target} (score: {best_candidate.confidence_model:.3f})")
            print(f"  Rationale: {best_candidate.rationale}")


def demonstrate_semantic_search(rag_service: RAGService):
    """Demonstrate semantic search functionality."""
    print("\n=== Semantic Search Demo ===")
    
    # Search by semantic hints
    hints = ["heart rate", "cardiac", "pulse", "bpm"]
    results = rag_service.search_semantic_hints(hints, top_k=5)
    
    print(f"Search Hints: {', '.join(hints)}")
    print(f"Results: {len(results)}")
    
    for i, doc in enumerate(results):
        print(f"\n{i+1}. {doc['table']}.{doc['column']}")
        print(f"   Description: {doc['metadata'].get('description', 'No description')}")
        print(f"   Type: {doc['metadata'].get('data_type', 'Unknown')}")
        if doc['metadata'].get('units'):
            print(f"   Units: {doc['metadata']['units']}")


def demonstrate_json_api_usage(rag_service: RAGService):
    """Demonstrate JSON API usage."""
    print("\n=== JSON API Usage Demo ===")
    
    # Create field data as JSON
    field_data = {
        "path": "patient.birth_date",
        "name_tokens": ["birth", "date", "dob"],
        "inferred_type": "datetime",
        "format_regex": "YYYY-MM-DD",
        "hints": ["date of birth", "patient age"],
        "coarse_semantics": ["temporal", "demographic"]
    }
    
    # Use JSON API
    result = rag_service.match_field_by_path(**field_data)
    
    print(f"JSON API Result for {field_data['path']}:")
    print(f"  Decision: {result['decision']['action']}")
    print(f"  Confidence: {result['decision']['final_confidence']:.3f}")
    
    if result['decision']['selected_target']:
        print(f"  Target: {result['decision']['selected_target']}")


def main():
    """Main demonstration function."""
    print("RAG Schema Matching System Demo")
    print("=" * 50)
    
    # Note: In a real implementation, you would initialize the RAG service
    # with actual components. For this demo, we'll show the expected usage.
    
    print("""
This demo shows how to use the RAG schema matching system.

To run this demo with actual functionality, you need to:

1. Set up the knowledge base with MIMIC-III schema:
   python scripts/setup_rag_system.py

2. Initialize the RAG service with proper components:
   - Vector store with MIMIC-III embeddings
   - Embedding service with bi-encoder and cross-encoder
   - LLM client (OpenAI, Anthropic, etc.)
   - Scoring system with calibrated thresholds

3. Run the matching operations as shown below.
""")
    
    # Example usage patterns
    print("\nExample Usage Patterns:")
    
    # Single field matching
    print("""
# Single field matching
source_field = SourceField(
    path="patient.heart_rate",
    name_tokens=["heart", "rate"],
    inferred_type=FieldType.INTEGER,
    units="bpm",
    hints=["vital signs", "cardiac"]
)

result = rag_service._orchestrator.match_single_field(source_field)
print(f"Decision: {result.decision.action}")
print(f"Target: {result.decision.selected_target}")
""")
    
    # Batch matching
    print("""
# Batch field matching
source_fields = [field1, field2, field3, ...]
results = rag_service._orchestrator.match_schema_fields(source_fields)

# Analyze results
for result in results:
    print(f"{result.source_field}: {result.decision.action}")
""")
    
    # JSON API usage
    print("""
# JSON API usage
field_data = {
    "path": "patient.id",
    "name_tokens": ["patient", "id"],
    "inferred_type": "id"
}

result = rag_service.match_field_by_path(**field_data)
""")
    
    # Semantic search
    print("""
# Semantic search
hints = ["heart rate", "cardiac", "vital signs"]
results = rag_service.search_semantic_hints(hints, top_k=10)
""")
    
    print("\nFor more examples, see the test files in tests/unit/ and tests/integration/")
    print("For API usage, see src/presentation/api/rag_endpoints.py")


if __name__ == "__main__":
    main()

