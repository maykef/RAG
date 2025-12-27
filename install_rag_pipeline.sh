#!/bin/bash
# =============================================================================
# Installation Script for Scientific RAG Pipeline (Mamba)
# =============================================================================
# Target Hardware:
#   - AMD Threadripper 7970X (32-core)
#   - NVIDIA RTX PRO 6000 96GB Blackwell
#   - 256GB RAM, 24TB NVMe
# =============================================================================

set -e

echo "=============================================="
echo "Scientific RAG Pipeline - Installation Script"
echo "=============================================="
echo ""

# Environment name
ENV_NAME="rag-pipeline"
PYTHON_VERSION="3.11"

# Check if mamba is available
if ! command -v mamba &> /dev/null; then
    echo "Error: mamba not found. Please install mamba/miniforge first:"
    echo ""
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  bash Miniforge3-Linux-x86_64.sh"
    echo ""
    exit 1
fi

echo "Using mamba: $(mamba --version)"

# Check CUDA availability
echo ""
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    echo "Warning: nvidia-smi not found. CUDA may not be properly configured."
fi

# =============================================================================
# Clean Up Existing Environment (if present)
# =============================================================================
echo ""
echo "Checking for existing environment: $ENV_NAME"

# Initialize mamba shell hook first (needed for deactivate to work)
eval "$(mamba shell hook --shell bash)"

# Check if environment exists - look for the env name anywhere in the line
# mamba env list format: "  name   Active  /path/to/env"
ENV_EXISTS=$(mamba env list | grep -w "$ENV_NAME" || true)

if [[ -n "$ENV_EXISTS" ]]; then
    echo "Found existing environment:"
    echo "  $ENV_EXISTS"
    echo ""
    echo "Performing clean removal..."
    
    # Deactivate if currently active
    if [[ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]]; then
        echo "Deactivating current environment..."
        mamba deactivate || conda deactivate || true
    fi
    
    # Get environment path - extract the path (last field that starts with /)
    ENV_PATH=$(echo "$ENV_EXISTS" | grep -oE '/[^ ]+$' | head -1)
    echo "Environment path: $ENV_PATH"
    
    # Unregister/remove any kernel specs if jupyter was installed
    if command -v jupyter &> /dev/null; then
        echo "Removing Jupyter kernel spec..."
        jupyter kernelspec uninstall $ENV_NAME -y 2>/dev/null || true
    fi
    
    # Remove the environment
    echo "Removing environment with mamba..."
    mamba env remove -n $ENV_NAME -y || {
        echo "mamba env remove failed, trying conda..."
        conda env remove -n $ENV_NAME -y || true
    }
    
    # Force remove directory if it still exists (edge cases)
    if [[ -n "$ENV_PATH" ]] && [[ -d "$ENV_PATH" ]]; then
        echo "Force removing residual environment directory: $ENV_PATH"
        rm -rf "$ENV_PATH"
    fi
    
    # Clean conda/mamba caches and orphaned packages
    echo "Cleaning package caches..."
    mamba clean --all -y 2>/dev/null || true
    
    echo "✓ Previous environment cleaned successfully"
else
    echo "No existing environment found. Proceeding with fresh install."
fi

# =============================================================================
# Create Mamba Environment
# =============================================================================
echo ""
echo "Creating mamba environment: $ENV_NAME"

# Create new environment with Python and CUDA toolkit 12.6 (Blackwell support)
mamba create -n $ENV_NAME \
    python=$PYTHON_VERSION \
    cuda-toolkit=12.6 \
    cudnn \
    -c conda-forge \
    -c nvidia \
    -y

# Activate environment
echo ""
echo "Activating environment..."

# Initialize mamba shell hook (required for activation in scripts)
eval "$(mamba shell hook --shell bash)"
mamba activate $ENV_NAME

echo "Python: $(python --version)"
echo "Environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Install PyTorch with CUDA Support (2.7+ with Blackwell sm_120 support)
# =============================================================================
echo ""
echo "Installing PyTorch 2.7+ with Blackwell (sm_120) support via CUDA 12.8..."

# Uninstall any existing PyTorch first
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch 2.7+ with CUDA 12.8 (official Blackwell support)
pip install --no-cache-dir \
    torch>=2.7.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# Install Core Dependencies via Mamba (where available)
# =============================================================================
echo ""
echo "Installing core dependencies via mamba..."

mamba install \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    pillow>=10.0.0 \
    requests>=2.31.0 \
    pyyaml>=6.0.0 \
    tqdm>=4.66.0 \
    -c conda-forge \
    -y

# =============================================================================
# Install Python Packages via pip (not available in conda)
# =============================================================================
echo ""
echo "Installing document parsing libraries via pip..."

pip install --no-cache-dir \
    docling>=2.0.0 \
    marker-pdf>=0.3.0 \
    pypdfium2>=4.0.0 \
    pdfplumber>=0.10.0 \
    pdf2image>=1.16.0

echo ""
echo "Installing embedding and vector search libraries..."

pip install --no-cache-dir \
    sentence-transformers>=2.7.0 \
    transformers>=4.40.0 \
    huggingface-hub>=0.23.0 \
    accelerate>=0.30.0

pip install --no-cache-dir \
    chromadb>=0.4.24 \
    qdrant-client>=1.8.0

echo ""
echo "Installing LangChain ecosystem..."

pip install --no-cache-dir \
    langchain>=0.1.0 \
    langchain-community>=0.0.20 \
    langchain-huggingface>=0.0.1

echo ""
echo "Installing additional utilities..."

pip install --no-cache-dir \
    httpx>=0.27.0 \
    aiohttp>=3.9.0 \
    rich>=13.0.0 \
    click>=8.1.0 \
    python-dotenv>=1.0.0

# =============================================================================
# Ollama Installation Check
# =============================================================================
echo ""
echo "Checking Ollama installation..."

if command -v ollama &> /dev/null; then
    echo "Ollama is installed: $(ollama --version)"
    
    echo ""
    echo "Pulling recommended LLM models..."
    
    # Pull Llama 3.1 70B (Q4_K_M quantization for 96GB VRAM)
    echo "Pulling llama3.1:70b-instruct-q4_K_M..."
    ollama pull llama3.1:70b-instruct-q4_K_M || echo "Note: Model pull failed or already exists"
    
    # Alternative: Qwen 2.5 72B
    echo "Pulling qwen2.5:72b-instruct-q4_K_M..."
    ollama pull qwen2.5:72b-instruct-q4_K_M || echo "Note: Model pull failed or already exists"
    
else
    echo ""
    echo "Ollama not found. Please install Ollama:"
    echo ""
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Then pull the recommended models:"
    echo "  ollama pull llama3.1:70b-instruct-q4_K_M"
    echo "  ollama pull qwen2.5:72b-instruct-q4_K_M"
fi

# =============================================================================
# Verify Installation
# =============================================================================
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python << 'EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        __import__(name)
        print(f"  ✓ {package}")
        return True
    except ImportError as e:
        print(f"  ✗ {package}: {e}")
        return False

print("\nCore Libraries:")
check_import("torch")
check_import("numpy")
check_import("pandas")

print("\nDocument Parsing:")
check_import("docling")
check_import("marker", "marker-pdf")
check_import("pypdfium2")

print("\nEmbeddings & Vector Stores:")
check_import("sentence_transformers", "sentence-transformers")
check_import("chromadb")
check_import("qdrant_client", "qdrant-client")

print("\nLangChain:")
check_import("langchain")
check_import("langchain_community")

print("\nCUDA Status:")
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ✗ CUDA not available")

print("")
EOF

# =============================================================================
# Create Configuration Directory
# =============================================================================
echo ""
echo "Setting up configuration..."

mkdir -p /mnt/nvme8tb/vector_store
mkdir -p /mnt/nvme8tb/cache
mkdir -p /mnt/nvme8tb/doc_cache
mkdir -p ./documents

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Environment: $ENV_NAME"
echo ""
echo "To activate the environment:"
echo "  mamba activate $ENV_NAME"
echo ""
echo "Next Steps:"
echo "1. Activate: mamba activate $ENV_NAME"
echo "2. Start Ollama: systemctl start ollama"
echo "3. Place PDFs in: ./documents/"
echo "4. Ingest: python scientific_rag_pipeline.py ingest ./documents/"
echo "5. Query: python scientific_rag_pipeline.py query 'Your question here'"
echo ""
echo "For interactive mode:"
echo "  python scientific_rag_pipeline.py interactive"
echo ""
