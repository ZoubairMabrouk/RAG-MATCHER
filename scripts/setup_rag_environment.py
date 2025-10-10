#!/usr/bin/env python3
"""
Setup script for RAG Virtual Rename environment.

This script helps configure the environment for the RAG-based virtual renaming system.
It handles:
- Package installation verification
- Environment variable configuration
- Basic system validation
- Demo data setup

Usage:
    python scripts/setup_rag_environment.py [--llm] [--api-key KEY]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_packages():
    """Check and install required packages."""
    print("🔍 Checking required packages...")
    
    packages = [
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("numpy", "numpy"),
        ("openai", "openai"),  # Optional
    ]
    
    missing_packages = []
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            print(f"✅ {package_name} is installed")
        else:
            print(f"❌ {package_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    
    print("✅ All required packages are available")
    return True


def setup_environment_variables(use_llm=False, api_key=None):
    """Setup environment variables."""
    print("\n🔧 Setting up environment variables...")
    
    # Set RAG_USE_LLM
    if use_llm:
        os.environ["RAG_USE_LLM"] = "1"
        print("✅ RAG_USE_LLM=1 (LLM validation enabled)")
        
        # Set API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ OPENAI_API_KEY set")
        else:
            existing_key = os.getenv("OPENAI_API_KEY")
            if existing_key:
                print("✅ OPENAI_API_KEY already set")
            else:
                print("⚠️  OPENAI_API_KEY not set - LLM validation will fail")
                print("   Set it manually: export OPENAI_API_KEY=your_key")
    else:
        os.environ["RAG_USE_LLM"] = "0"
        print("✅ RAG_USE_LLM=0 (retrieval-only mode)")
    
    # Set embedding provider
    os.environ["RAG_EMBED_PROVIDER"] = "local"
    print("✅ RAG_EMBED_PROVIDER=local")
    
    # Set local embedding model
    os.environ["LOCAL_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    print("✅ LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2")


def create_demo_config():
    """Create demo configuration files."""
    print("\n📝 Creating demo configuration...")
    
    demo_dir = Path("examples/demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # Create demo schema file
    demo_schema = """{
  "tables": [
    {
      "name": "products",
      "columns": [
        {"name": "id", "data_type": "INTEGER", "is_primary_key": true, "is_nullable": false},
        {"name": "name", "data_type": "VARCHAR(255)", "is_nullable": false},
        {"name": "price", "data_type": "DECIMAL(10,2)", "is_nullable": false},
        {"name": "quantity", "data_type": "INTEGER", "is_nullable": true},
        {"name": "reference", "data_type": "VARCHAR(255)", "is_nullable": true}
      ]
    },
    {
      "name": "customers",
      "columns": [
        {"name": "id", "data_type": "INTEGER", "is_primary_key": true, "is_nullable": false},
        {"name": "email", "data_type": "VARCHAR(255)", "is_nullable": false},
        {"name": "first_name", "data_type": "VARCHAR(100)", "is_nullable": false},
        {"name": "last_name", "data_type": "VARCHAR(100)", "is_nullable": false}
      ]
    }
  ]
}"""
    
    with open(demo_dir / "current_schema.json", "w") as f:
        f.write(demo_schema)
    
    # Create demo U-Schema file
    demo_uschema = """{
  "entities": [
    {
      "name": "items",
      "attributes": [
        {"name": "id", "data_type": "INTEGER", "required": true},
        {"name": "name", "data_type": "STRING", "required": true},
        {"name": "price", "data_type": "DECIMAL", "required": true},
        {"name": "qte", "data_type": "INTEGER", "required": false},
        {"name": "ref", "data_type": "STRING", "required": false}
      ]
    },
    {
      "name": "users",
      "attributes": [
        {"name": "id", "data_type": "INTEGER", "required": true},
        {"name": "email", "data_type": "STRING", "required": true},
        {"name": "firstName", "data_type": "STRING", "required": true},
        {"name": "lastName", "data_type": "STRING", "required": true}
      ]
    }
  ]
}"""
    
    with open(demo_dir / "uschema.json", "w") as f:
        f.write(demo_uschema)
    
    print(f"✅ Demo data created in {demo_dir}")
    print("   - current_schema.json")
    print("   - uschema.json")


def test_basic_functionality():
    """Test basic RAG functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from src.infrastructure.rag.embedding_service import LocalEmbeddingProvider
        from src.infrastructure.rag.vector_store import RAGVectorStore
        from src.infrastructure.rag.rag_schema_matcher import RAGSchemaMatcher
        
        # Test embedding provider
        provider = LocalEmbeddingProvider()
        embeddings = provider.embed(["test text"])
        print(f"✅ Embedding provider working (dimension: {provider.dimension})")
        
        # Test vector store
        vector_store = RAGVectorStore(dimension=provider.dimension)
        print("✅ Vector store created")
        
        # Test RAG matcher
        from src.infrastructure.rag.embedding_service import EmbeddingService
        embedding_service = EmbeddingService(provider)
        matcher = RAGSchemaMatcher(embedding_service, vector_store)
        print("✅ RAG schema matcher created")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def print_next_steps(use_llm=False):
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    
    if use_llm:
        print("1. Set your OpenAI API key:")
        print("   export OPENAI_API_KEY=your_api_key_here")
        print("   or add it to your .env file")
    
    print("2. Run the validation script:")
    print("   python scripts/validate_rag_implementation.py")
    
    print("3. Try the demo:")
    print("   python examples/run_rag_virtual_rename_demo.py")
    
    print("4. Run the test suite:")
    print("   python examples/test_rag_virtual_rename.py")
    
    print("\n📚 Documentation:")
    print("   - README: docs/RAG_VIRTUAL_RENAME.md")
    print("   - Examples: examples/")
    print("   - Scripts: scripts/")
    
    print("\n🔧 Configuration:")
    print("   - RAG_USE_LLM=0 (retrieval-only) or RAG_USE_LLM=1 (with LLM)")
    print("   - RAG_EMBED_PROVIDER=local (default) or openai")
    print("   - LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2 (default)")
    
    print("\n💡 Tips:")
    print("   - Start with retrieval-only mode for testing")
    print("   - Enable LLM validation for production use")
    print("   - Adjust thresholds in RAGSchemaMatcher if needed")
    print("   - Check logs for detailed matching information")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup RAG Virtual Rename environment")
    parser.add_argument("--llm", action="store_true", help="Enable LLM validation")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--skip-packages", action="store_true", help="Skip package installation")
    parser.add_argument("--skip-demo", action="store_true", help="Skip demo data creation")
    
    args = parser.parse_args()
    
    print("🚀 RAG Virtual Rename Environment Setup")
    print("="*50)
    
    # Check packages
    if not args.skip_packages:
        if not check_and_install_packages():
            print("❌ Package setup failed")
            return 1
    else:
        print("⏭️  Skipping package installation")
    
    # Setup environment
    setup_environment_variables(use_llm=args.llm, api_key=args.api_key)
    
    # Create demo data
    if not args.skip_demo:
        create_demo_config()
    else:
        print("⏭️  Skipping demo data creation")
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality test failed")
        return 1
    
    # Print next steps
    print_next_steps(use_llm=args.llm)
    
    return 0


if __name__ == "__main__":
    exit(main())
