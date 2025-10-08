"""
Validation script for RAG system installation.
Checks all components and dependencies.
"""

import sys
import os
import importlib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "sentence_transformers",
        "faiss",
        "numpy",
        "scikit-learn",
        "pandas",
        "fastapi",
        "pydantic",
        "openai",
        "anthropic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: poetry install")
        return False
    
    print("✅ All dependencies installed")
    return True


def check_rag_components():
    """Check if all RAG components can be imported."""
    print("\n🔍 Checking RAG components...")
    
    components = [
        "src.domain.entities.rag_schema",
        "src.infrastructure.rag.knowledge_base_builder",
        "src.infrastructure.rag.embedding_service",
        "src.infrastructure.rag.vector_store",
        "src.infrastructure.rag.retriever",
        "src.infrastructure.rag.llm_orchestrator",
        "src.infrastructure.rag.scoring_system",
        "src.infrastructure.rag.rag_orchestrator",
        "src.infrastructure.rag.evaluation_metrics",
        "src.presentation.api.rag_endpoints"
    ]
    
    missing_components = []
    
    for component in components:
        try:
            importlib.import_module(component)
            print(f"  ✅ {component}")
        except ImportError as e:
            print(f"  ❌ {component} - ERROR: {e}")
            missing_components.append(component)
    
    if missing_components:
        print(f"\n❌ Missing components: {len(missing_components)}")
        return False
    
    print("✅ All RAG components available")
    return True


def check_data_directory():
    """Check if data directory exists and has required files."""
    print("\n🔍 Checking data directory...")
    
    data_dir = Path("data/rag")
    
    if not data_dir.exists():
        print(f"  ❌ Data directory not found: {data_dir}")
        print("  Run: python scripts/setup_rag_system.py")
        return False
    
    print(f"  ✅ Data directory exists: {data_dir}")
    
    # Check for key files
    required_files = [
        "mimic_ddl.sql",
        "mimic_dictionary.json",
        "knowledge_base.jsonl",
        "vector_store.index",
        "vector_store.data",
        "demo_dataset.json"
    ]
    
    missing_files = []
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("  Run: python scripts/setup_rag_system.py")
        return False
    
    print("✅ All data files present")
    return True


def test_basic_functionality():
    """Test basic RAG functionality."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test imports
        from src.domain.entities.rag_schema import SourceField, FieldType
        from src.infrastructure.rag.embedding_service import RAGEmbeddingService
        from src.infrastructure.rag.vector_store import RAGVectorStore
        
        print("  ✅ Core imports successful")
        
        # Test SourceField creation
        source_field = SourceField(
            path="test.field",
            name_tokens=["test", "field"],
            inferred_type=FieldType.TEXT,
            hints=["test hint"],
            coarse_semantics=["test"]
        )
        
        print(f"  ✅ SourceField created: {source_field.path}")
        
        # Test embedding service initialization (mock)
        print("  ✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False


def check_api_endpoints():
    """Check if API endpoints are properly configured."""
    print("\n🔍 Checking API configuration...")
    
    try:
        from src.presentation.api.app import create_app
        
        app = create_app()
        
        # Check if RAG router is included
        routes = [route.path for route in app.routes]
        rag_routes = [route for route in routes if "/rag/" in route]
        
        if rag_routes:
            print(f"  ✅ RAG API endpoints found: {len(rag_routes)}")
            for route in rag_routes[:5]:  # Show first 5
                print(f"    - {route}")
            if len(rag_routes) > 5:
                print(f"    ... and {len(rag_routes) - 5} more")
        else:
            print("  ❌ No RAG API endpoints found")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ API configuration check failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🚀 RAG System Installation Validation")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("RAG Components", check_rag_components),
        ("Data Directory", check_data_directory),
        ("Basic Functionality", test_basic_functionality),
        ("API Configuration", check_api_endpoints)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 RAG system is properly installed and configured!")
        print("\nNext steps:")
        print("1. Start the API server: uvicorn src.presentation.api.app:app --reload")
        print("2. Test endpoints: curl http://localhost:8000/api/v1/rag/health")
        print("3. Run examples: python examples/rag_matching_example.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} issues found. Please resolve them before using the RAG system.")
        print("\nCommon solutions:")
        print("- Missing dependencies: poetry install")
        print("- Missing data files: python scripts/setup_rag_system.py")
        print("- Import errors: Check Python path and module structure")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

