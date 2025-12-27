#!/usr/bin/env python3
"""
RAG Pipeline Installation Test
==============================
Comprehensive verification of all components.
"""

import sys
import os

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_result(name, success, details=""):
    status = "✓" if success else "✗"
    print(f"  {status} {name}")
    if details:
        print(f"      {details}")

def test_core_libraries():
    print_header("Core Libraries")
    
    # PyTorch
    try:
        import torch
        print_result("PyTorch", True, f"v{torch.__version__}")
    except ImportError as e:
        print_result("PyTorch", False, str(e))
        return False
    
    # NumPy
    try:
        import numpy as np
        print_result("NumPy", True, f"v{np.__version__}")
    except ImportError as e:
        print_result("NumPy", False, str(e))
    
    # Pandas
    try:
        import pandas as pd
        print_result("Pandas", True, f"v{pd.__version__}")
    except ImportError as e:
        print_result("Pandas", False, str(e))
    
    return True

def test_cuda():
    print_header("CUDA / GPU Status")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print_result("CUDA Available", cuda_available)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print_result("GPU Count", True, str(device_count))
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                print_result(f"GPU {i}", True, f"{name} ({vram_gb:.1f} GB VRAM)")
            
            # Test CUDA computation
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                print_result("CUDA Computation Test", True, "Matrix multiplication successful")
            except Exception as e:
                print_result("CUDA Computation Test", False, str(e))
            
            # cuDNN
            cudnn_available = torch.backends.cudnn.is_available()
            cudnn_version = torch.backends.cudnn.version() if cudnn_available else "N/A"
            print_result("cuDNN", cudnn_available, f"v{cudnn_version}" if cudnn_available else "Not available")
            
            return True
        else:
            print("      CUDA not available - checking why...")
            print(f"      torch.version.cuda: {torch.version.cuda}")
            print(f"      CUDA_HOME env: {os.environ.get('CUDA_HOME', 'Not set')}")
            return False
            
    except Exception as e:
        print_result("CUDA Test", False, str(e))
        return False

def test_document_parsing():
    print_header("Document Parsing Libraries")
    
    # Docling
    try:
        import docling
        print_result("Docling", True, f"v{docling.__version__}" if hasattr(docling, '__version__') else "installed")
    except ImportError as e:
        print_result("Docling", False, str(e))
    
    # Marker
    try:
        import marker
        print_result("Marker", True, "installed")
    except ImportError as e:
        print_result("Marker", False, str(e))
    
    # pypdfium2
    try:
        import pypdfium2
        print_result("pypdfium2", True, f"v{pypdfium2.__version__}" if hasattr(pypdfium2, '__version__') else "installed")
    except ImportError as e:
        print_result("pypdfium2", False, str(e))
    
    # pdfplumber
    try:
        import pdfplumber
        print_result("pdfplumber", True, f"v{pdfplumber.__version__}" if hasattr(pdfplumber, '__version__') else "installed")
    except ImportError as e:
        print_result("pdfplumber", False, str(e))

def test_embeddings():
    print_header("Embedding Libraries")
    
    # Sentence Transformers
    try:
        import sentence_transformers
        print_result("sentence-transformers", True, f"v{sentence_transformers.__version__}")
    except ImportError as e:
        print_result("sentence-transformers", False, str(e))
        return
    
    # Transformers
    try:
        import transformers
        print_result("transformers", True, f"v{transformers.__version__}")
    except ImportError as e:
        print_result("transformers", False, str(e))
    
    # HuggingFace Hub
    try:
        import huggingface_hub
        print_result("huggingface-hub", True, f"v{huggingface_hub.__version__}")
    except ImportError as e:
        print_result("huggingface-hub", False, str(e))

def test_vector_stores():
    print_header("Vector Store Libraries")
    
    # ChromaDB
    try:
        import chromadb
        print_result("ChromaDB", True, f"v{chromadb.__version__}")
    except ImportError as e:
        print_result("ChromaDB", False, str(e))
    
    # Qdrant
    try:
        import qdrant_client
        print_result("Qdrant Client", True, f"v{qdrant_client.__version__}" if hasattr(qdrant_client, '__version__') else "installed")
    except ImportError as e:
        print_result("Qdrant Client", False, str(e))

def test_langchain():
    print_header("LangChain Ecosystem")
    
    try:
        import langchain
        print_result("langchain", True, f"v{langchain.__version__}")
    except ImportError as e:
        print_result("langchain", False, str(e))
    
    try:
        import langchain_community
        print_result("langchain-community", True, "installed")
    except ImportError as e:
        print_result("langchain-community", False, str(e))
    
    try:
        import langchain_huggingface
        print_result("langchain-huggingface", True, "installed")
    except ImportError as e:
        print_result("langchain-huggingface", False, str(e))

def test_storage_paths():
    print_header("Storage Paths")
    
    paths = [
        "/mnt/nvme8tb",
        "/mnt/nvme8tb/vector_store",
        "/mnt/nvme8tb/cache",
        "/mnt/nvme8tb/doc_cache",
    ]
    
    for path in paths:
        exists = os.path.exists(path)
        writable = os.access(path, os.W_OK) if exists else False
        
        if exists and writable:
            print_result(path, True, "exists, writable")
        elif exists:
            print_result(path, False, "exists but NOT writable")
        else:
            print_result(path, False, "does not exist")

def test_ollama():
    print_header("Ollama LLM Server")
    
    import subprocess
    import shutil
    
    # Check if ollama is installed
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print_result("Ollama Binary", False, "not found in PATH")
        return
    
    print_result("Ollama Binary", True, ollama_path)
    
    # Check ollama version
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        version = result.stdout.strip() or result.stderr.strip()
        print_result("Ollama Version", True, version)
    except Exception as e:
        print_result("Ollama Version", False, str(e))
    
    # Check if ollama server is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print_result("Ollama Server", True, "running")
            
            # List available models
            if models:
                print("\n  Available Models:")
                for model in models:
                    name = model.get("name", "unknown")
                    size_gb = model.get("size", 0) / 1e9
                    print(f"      - {name} ({size_gb:.1f} GB)")
            else:
                print("      No models installed. Run: ollama pull llama3.1:70b-instruct-q4_K_M")
        else:
            print_result("Ollama Server", False, f"HTTP {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_result("Ollama Server", False, "not running (start with: systemctl start ollama)")
    except Exception as e:
        print_result("Ollama Server", False, str(e))

def test_embedding_model_load():
    print_header("Embedding Model Test (Optional - may take time)")
    
    response = input("  Load and test embedding model? (y/N): ").strip().lower()
    if response != 'y':
        print("  Skipped.")
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading nomic-embed-text-v1.5 on {device}...")
        
        model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            device=device,
            trust_remote_code=True,
        )
        
        # Test embedding
        test_texts = [
            "search_document: This is a test scientific document about protein folding.",
            "search_query: What is protein folding?",
        ]
        
        embeddings = model.encode(test_texts, normalize_embeddings=True)
        
        print_result("Model Loaded", True, f"on {device}")
        print_result("Embedding Shape", True, str(embeddings.shape))
        
        # Check similarity
        import numpy as np
        similarity = np.dot(embeddings[0], embeddings[1])
        print_result("Similarity Score", True, f"{similarity:.4f}")
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        print_result("Embedding Test", False, str(e))

def main():
    print("\n" + "="*60)
    print(" RAG Pipeline Installation Verification")
    print("="*60)
    
    # Run all tests
    test_core_libraries()
    cuda_ok = test_cuda()
    test_document_parsing()
    test_embeddings()
    test_vector_stores()
    test_langchain()
    test_storage_paths()
    test_ollama()
    
    # Optional intensive test
    if cuda_ok:
        test_embedding_model_load()
    
    # Summary
    print_header("Summary")
    if cuda_ok:
        print("  ✓ CUDA is working - GPU acceleration available")
        print("  Ready to run the RAG pipeline!")
    else:
        print("  ✗ CUDA not available - will run on CPU (slower)")
        print("  Check PyTorch CUDA installation")
    
    print("\n  Next steps:")
    print("    1. Ensure Ollama is running: systemctl start ollama")
    print("    2. Pull an LLM: ollama pull llama3.1:70b-instruct-q4_K_M")
    print("    3. Place PDFs in ./documents/")
    print("    4. Run: python scientific_rag_pipeline.py ingest ./documents/")
    print("")

if __name__ == "__main__":
    main()
