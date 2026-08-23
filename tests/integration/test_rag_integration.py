"""
Integration tests for RAG schema matching system.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from src.domain.entities.rag_schema import (
    SourceField, FieldType, ScoringWeights, ScoringThresholds
)


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def generate_response(self, system_prompt: str, user_prompt: str, 
                         temperature: float = 0.0, max_tokens: int = 1024,
                         response_format: dict = None) -> str:
        """Generate mock response."""
        return json.dumps({
            "source_field": "test.field",
            "candidates": [
                {
                    "target": "TEST.TABLE.COLUMN",
                    "confidence_model": 0.85,
                    "confidence_llm": 0.82,
                    "rationale": "Test matching rationale"
                }
            ],
            "decision": {
                "action": "ACCEPT",
                "selected_target": "TEST.TABLE.COLUMN",
                "final_confidence": 0.84,
                "guardrails": ["type:test", "table:TEST"]
            }
        })


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_mimic_ddl():
    """Sample MIMIC-III DDL for testing."""
    return """
CREATE TABLE PATIENTS (
    SUBJECT_ID INTEGER PRIMARY KEY,
    GENDER VARCHAR(5),
    DOB DATE
);

CREATE TABLE ADMISSIONS (
    SUBJECT_ID INTEGER NOT NULL,
    HADM_ID INTEGER PRIMARY KEY,
    ADMITTIME TIMESTAMP,
    FOREIGN KEY (SUBJECT_ID) REFERENCES PATIENTS(SUBJECT_ID)
);
"""


@pytest.fixture
def sample_dictionary():
    """Sample data dictionary for testing."""
    return {
        "PATIENTS.SUBJECT_ID": {
            "description": "Unique patient identifier",
            "synonyms": ["patient_id"],
            "units": None
        },
        "ADMISSIONS.ADMITTIME": {
            "description": "Admission timestamp",
            "synonyms": ["admission_time"],
            "units": "timestamp"
        }
    }


@pytest.fixture
def sample_source_field():
    """Sample source field for testing."""
    return SourceField(
        path="patient.id",
        name_tokens=["patient", "id"],
        inferred_type=FieldType.ID,
        hints=["patient identifier"],
        coarse_semantics=["identifier"]
    )


class TestKnowledgeBaseBuilder:
    """Test knowledge base builder."""
    
    def test_build_from_ddl(self, temp_data_dir, sample_mimic_ddl, sample_dictionary):
        """Test building knowledge base from DDL and dictionary."""
        from src.infrastructure.rag.knowledge_base_builder import MIMICKnowledgeBaseBuilder
        
        # Write test files
        ddl_path = os.path.join(temp_data_dir, "test_ddl.sql")
        dict_path = os.path.join(temp_data_dir, "test_dict.json")
        
        with open(ddl_path, 'w') as f:
            f.write(sample_mimic_ddl)
        
        with open(dict_path, 'w') as f:
            json.dump(sample_dictionary, f)
        
        # Build knowledge base
        builder = MIMICKnowledgeBaseBuilder()
        documents = builder.build_from_ddl(ddl_path, dict_path)
        
        # Verify results
        assert len(documents) > 0
        assert any(doc.table == "PATIENTS" for doc in documents)
        assert any(doc.table == "ADMISSIONS" for doc in documents)
        
        # Verify document structure
        for doc in documents:
            assert doc.id is not None
            assert doc.table is not None
            assert doc.column is not None
            assert doc.content is not None
            assert isinstance(doc.metadata, dict)
    
    def test_save_and_load_documents(self, temp_data_dir):
        """Test saving and loading documents."""
        from src.domain.entities.rag_schema import KnowledgeBaseDocument
        from src.infrastructure.rag.knowledge_base_builder import MIMICKnowledgeBaseBuilder
        
        docs = [
            KnowledgeBaseDocument(
                id="TEST.TABLE1.COL1",
                table="TABLE1",
                column="COL1",
                content="Test content 1",
                metadata={"type": "test"}
            ),
            KnowledgeBaseDocument(
                id="TEST.TABLE2.COL2",
                table="TABLE2",
                column="COL2",
                content="Test content 2",
                metadata={"type": "test"}
            )
        ]
        
        # Save documents
        builder = MIMICKnowledgeBaseBuilder()
        output_path = os.path.join(temp_data_dir, "test_docs.jsonl")
        builder.save_documents(docs, output_path)
        
        # Load documents
        loaded_docs = builder.load_documents(output_path)
        
        # Verify
        assert len(loaded_docs) == len(docs)
        assert loaded_docs[0].id == docs[0].id
        assert loaded_docs[1].id == docs[1].id


class TestEmbeddingService:
    """Test embedding service."""
    
    @patch('src.infrastructure.rag.embedding_service.SentenceTransformer')
    @patch('src.infrastructure.rag.embedding_service.CrossEncoder')
    def test_embedding_service_initialization(self, mock_cross, mock_bi):
        """Test embedding service initialization."""
        from src.infrastructure.rag.embedding_service import RAGEmbeddingService
        
        # Configure mocks
        mock_bi_instance = MagicMock()
        mock_bi_instance.get_sentence_embedding_dimension.return_value = 384
        mock_bi.return_value = mock_bi_instance
        
        mock_cross_instance = MagicMock()
        mock_cross.return_value = mock_cross_instance
        
        # Create service
        service = RAGEmbeddingService()
        
        # Verify initialization
        assert mock_bi.called
        assert mock_cross.called
        assert service.dimension == 384
    
    @patch('src.infrastructure.rag.embedding_service.SentenceTransformer')
    @patch('src.infrastructure.rag.embedding_service.CrossEncoder')
    def test_embed_documents(self, mock_cross, mock_bi):
        """Test document embedding."""
        from src.domain.entities.rag_schema import KnowledgeBaseDocument
        from src.infrastructure.rag.embedding_service import RAGEmbeddingService
        import numpy as np
        
        docs = [
            KnowledgeBaseDocument(
                id="TEST.COL1",
                table="TEST",
                column="COL1",
                content="Test content",
                metadata={}
            )
        ]
        
        # Configure mocks
        mock_bi_instance = MagicMock()
        mock_bi_instance.get_sentence_embedding_dimension.return_value = 384
        mock_bi_instance.encode.return_value = np.array([[0.1] * 384])
        mock_bi.return_value = mock_bi_instance
        
        mock_cross_instance = MagicMock()
        mock_cross.return_value = mock_cross_instance
        
        # Create service and test embedding
        service = RAGEmbeddingService()
        embeddings = service.embed_documents(docs)
        
        assert embeddings.shape == (1, 384)
        assert mock_bi_instance.encode.called


class TestVectorStore:
    """Test vector store."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        from src.infrastructure.rag.vector_store import RAGVectorStore
        
        store = RAGVectorStore(dimension=384, index_type="Flat")
        assert store._dimension == 384
        assert len(store._documents) == 0
    
    def test_add_documents(self):
        """Test adding documents to vector store."""
        from src.domain.entities.rag_schema import KnowledgeBaseDocument
        from src.infrastructure.rag.vector_store import RAGVectorStore
        import numpy as np
        
        docs = [
            KnowledgeBaseDocument(
                id="TEST.COL1",
                table="TEST",
                column="COL1",
                content="Test content",
                metadata={}
            )
        ]
        
        embeddings = np.array([[0.1] * 384])
        
        store = RAGVectorStore(dimension=384, index_type="Flat")
        store.add_documents(docs, embeddings)
        
        assert len(store._documents) == 1
        assert store._embeddings.shape == (1, 384)
    
    def test_search(self):
        """Test vector store search."""
        from src.domain.entities.rag_schema import KnowledgeBaseDocument
        from src.infrastructure.rag.vector_store import RAGVectorStore
        import numpy as np
        
        docs = [
            KnowledgeBaseDocument(
                id="TEST.COL1",
                table="TEST",
                column="COL1",
                content="Test content",
                metadata={"type": "test"}
            )
        ]
        
        embeddings = np.array([[1.0, 0.0, 0.0]])  # Simple embedding
        
        store = RAGVectorStore(dimension=3, index_type="Flat")
        store.add_documents(docs, embeddings)
        
        # Search with similar embedding
        query_embedding = np.array([0.9, 0.1, 0.0])
        results = store.search(query_embedding, top_k=1)
        
        assert len(results) == 1
        assert results[0][0].id == "TEST.COL1"


class TestRAGOrchestrator:
    """Test RAG orchestrator integration."""
    
    @patch('src.infrastructure.rag.embedding_service.SentenceTransformer')
    @patch('src.infrastructure.rag.embedding_service.CrossEncoder')
    def test_orchestrator_initialization(self, mock_cross, mock_bi):
        """Test orchestrator initialization."""
        from src.infrastructure.rag.embedding_service import RAGEmbeddingService
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_orchestrator import RAGOrchestrator
        
        # Configure mocks
        mock_bi_instance = MagicMock()
        mock_bi_instance.get_sentence_embedding_dimension.return_value = 384
        mock_bi.return_value = mock_bi_instance
        
        mock_cross_instance = MagicMock()
        mock_cross.return_value = mock_cross_instance
        
        # Create components
        embedding_service = RAGEmbeddingService()
        vector_store = RAGVectorStore(dimension=384)
        llm_client = MockLLMClient()
        
        orchestrator = RAGOrchestrator(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client
        )
        
        assert orchestrator._vector_store is not None
        assert orchestrator._embedding_service is not None
        assert orchestrator._retriever is not None
        assert orchestrator._llm_orchestrator is not None
        assert orchestrator._scoring_system is not None
    
    @patch('src.infrastructure.rag.embedding_service.SentenceTransformer')
    @patch('src.infrastructure.rag.embedding_service.CrossEncoder')
    def test_match_single_field(self, mock_cross, mock_bi, sample_source_field):
        """Test single field matching."""
        from src.domain.entities.rag_schema import KnowledgeBaseDocument
        from src.infrastructure.rag.embedding_service import RAGEmbeddingService
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_orchestrator import RAGOrchestrator
        import numpy as np
        
        # Configure mocks
        mock_bi_instance = MagicMock()
        mock_bi_instance.get_sentence_embedding_dimension.return_value = 384
        mock_bi.return_value = mock_bi_instance
        
        mock_cross_instance = MagicMock()
        mock_cross.return_value = mock_cross_instance
        
        # Create components with mock data
        embedding_service = RAGEmbeddingService()
        
        # Use Flat index instead of IVFPQ to avoid training requirement
        vector_store = RAGVectorStore(dimension=384, index_type="Flat")
        
        # Add mock document to vector store
        mock_doc = KnowledgeBaseDocument(
            id="TEST.PATIENTS.SUBJECT_ID",
            table="PATIENTS",
            column="SUBJECT_ID",
            content="Patient identifier",
            metadata={"data_type": "integer"}
        )
        
        vector_store.add_documents([mock_doc], np.array([[0.1] * 384]))
        
        llm_client = MockLLMClient()
        
        orchestrator = RAGOrchestrator(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client
        )
        
        # Test matching
        result = orchestrator.match_single_field(sample_source_field)
        
        assert result.source_field == sample_source_field.path
        assert result.decision.action is not None
        assert result.processing_time_ms >= 0


class TestEvaluationDataset:
    """Test evaluation dataset."""
    
    def test_create_synthetic_dataset(self):
        """Test creating synthetic dataset."""
        from src.infrastructure.rag.evaluation_metrics import EvaluationDataset
        
        dataset = EvaluationDataset()
        dataset.create_synthetic_dataset(num_cases=10)
        
        assert len(dataset.ground_truth) == 10
        assert len(dataset.test_cases) == 10
        
        # Verify ground truth structure
        for gt in dataset.ground_truth:
            assert "source_field" in gt
            assert "correct_target" in gt
            assert "field_type" in gt
            assert "should_accept" in gt
    
    def test_add_ground_truth(self):
        """Test adding ground truth entries."""
        from src.infrastructure.rag.evaluation_metrics import EvaluationDataset
        
        dataset = EvaluationDataset()
        dataset.add_ground_truth(
            source_field="test.field",
            correct_target="TEST.TABLE.COLUMN",
            field_type="integer",
            should_accept=True,
            metadata={"test": "data"}
        )
        
        assert len(dataset.ground_truth) == 1
        assert dataset.ground_truth[0]["source_field"] == "test.field"
        assert dataset.ground_truth[0]["correct_target"] == "TEST.TABLE.COLUMN"
    
    def test_get_statistics(self):
        """Test dataset statistics."""
        from src.infrastructure.rag.evaluation_metrics import EvaluationDataset
        
        dataset = EvaluationDataset()
        dataset.create_synthetic_dataset(num_cases=5)
        
        stats = dataset.get_statistics()
        
        assert "total_ground_truth" in stats
        assert "total_test_cases" in stats
        assert "field_type_distribution" in stats
        assert "accept_rate" in stats
        
        assert stats["total_ground_truth"] == 5
        assert stats["total_test_cases"] == 5


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_evaluate_results(self):
        """Test evaluation of matching results."""
        from src.domain.entities.rag_schema import (
            SchemaMatchingResult, MatchingDecision, DecisionAction, CandidateMatch
        )
        from src.infrastructure.rag.evaluation_metrics import EvaluationMetrics
        
        # Create mock results
        results = [
            SchemaMatchingResult(
                source_field="test.field1",
                candidates=[
                    CandidateMatch(
                        target="TEST.TABLE.COLUMN",
                        confidence_model=0.85,
                        confidence_llm=0.82,
                        rationale="Test rationale"
                    )
                ],
                decision=MatchingDecision(
                    action=DecisionAction.ACCEPT,
                    selected_target="TEST.TABLE.COLUMN",
                    final_confidence=0.85,
                    guardrails=[]
                ),
                processing_time_ms=100.0,
                model_version="1.0"
            )
        ]
        
        # Create mock ground truth
        ground_truth = [
            {
                "source_field": "test.field1",
                "correct_target": "TEST.TABLE.COLUMN",
                "field_type": "integer",
                "should_accept": True
            }
        ]
        
        # Evaluate
        metrics = EvaluationMetrics()
        evaluation = metrics.evaluate_results(results, ground_truth)
        
        # Verify metrics structure
        assert "accuracy" in evaluation
        assert "precision_recall" in evaluation
        assert "coverage" in evaluation
        assert "efficiency" in evaluation
        assert "calibration" in evaluation
        assert "summary" in evaluation
        
        # Verify accuracy metrics
        assert "exact_match_accuracy" in evaluation["accuracy"]
        assert "top1_accuracy" in evaluation["accuracy"]
        assert "top5_accuracy" in evaluation["accuracy"]
        
        # Verify precision/recall metrics
        assert "precision" in evaluation["precision_recall"]
        assert "recall" in evaluation["precision_recall"]
        assert "f1_score" in evaluation["precision_recall"]


class TestRAGService:
    """Test RAG service high-level interface."""
    
    @patch('src.infrastructure.rag.embedding_service.SentenceTransformer')
    @patch('src.infrastructure.rag.embedding_service.CrossEncoder')
    def test_match_field_by_path(self, mock_cross, mock_bi):
        """Test matching field by path using high-level interface."""
        from src.infrastructure.rag.embedding_service import RAGEmbeddingService
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_orchestrator import RAGOrchestrator, RAGService
        
        # Configure mocks
        mock_bi_instance = MagicMock()
        mock_bi_instance.get_sentence_embedding_dimension.return_value = 384
        mock_bi.return_value = mock_bi_instance
        
        mock_cross_instance = MagicMock()
        mock_cross.return_value = mock_cross_instance
        
        # Create components
        embedding_service = RAGEmbeddingService()
        vector_store = RAGVectorStore(dimension=384)
        llm_client = MockLLMClient()
        
        orchestrator = RAGOrchestrator(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client
        )
        
        service = RAGService(orchestrator)
        
        # Test matching by path
        result = service.match_field_by_path(
            field_path="patient.id",
            name_tokens=["patient", "id"],
            inferred_type="id"
        )
        
        assert "source_field" in result
        assert "decision" in result
        assert "candidates" in result
        assert result["source_field"] == "patient.id"


if __name__ == "__main__":
    pytest.main([__file__])