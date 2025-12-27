Scientific RAG Pipeline
A production-ready, high-performance local Retrieval-Augmented Generation (RAG) pipeline optimized for complex scientific documents with multi-column layouts, tables, and mathematical formulas.
🎯 Target Hardware
Component	Specification
CPU	AMD Threadripper 7970X (32-core)
GPU	NVIDIA RTX PRO 6000 96GB Blackwell
RAM	256GB ECC RDIMM
Storage	24TB NVMe Scratch Pool (ZFS)
OS	Ubuntu 24.04 LTS with CUDA 12.6+

✨ Key Features
·	Vision-Based Document Parsing: Uses Docling (IBM) with GPU-accelerated table detection and OCR
·	Markdown-Aware Chunking: Respects document structure (headers, sections) for semantic coherence
·	High-Quality Embeddings: nomic-embed-text-v1.5 (8192 context, 768 dimensions)
·	Persistent Vector Storage: ChromaDB on NVMe for fast similarity search
·	Large LLM Support: Runs Llama 3.1 70B or Qwen 2.5 72B locally via Ollama
·	Explicit VRAM Management: Stage-based model loading/unloading to avoid OOM
·	Verification Tools: Built-in chunk inspection and quality verification
📊 Performance
Tested with 19 scientific PDFs:
Metric	Result
Ingestion Speed	19 PDFs → 2,608 chunks in ~3.5 minutes
Query Latency	15-40 seconds (embedding + retrieval + LLM)
Chunk Size	Mean 862 chars (512-1024 range: 83%)
VRAM Usage	0.55GB embeddings, ~50GB LLM

🚀 Quick Start
1. Installation
# Clone the repository
git clone https://github.com/yourusername/scientific-rag-pipeline.git
cd scientific-rag-pipeline

# Run the installation script
chmod +x install_rag_pipeline.sh
./install_rag_pipeline.sh

Or install manually:
# Create conda environment
conda create -n rag-pipeline python=3.11 -y
conda activate rag-pipeline

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:70b

2. Configure Storage Paths
Edit config.yaml to set your NVMe paths:
storage:
  scratch_base: /mnt/nvme8tb
  vector_db_path: /mnt/nvme8tb/vector_store
  cache_path: /mnt/nvme8tb/cache

3. Ingest Documents
# Single document
python scientific_rag_pipeline.py ingest paper.pdf

# Entire directory
python scientific_rag_pipeline.py ingest ./documents/ --extensions .pdf

4. Query
# Start Ollama first
sudo systemctl start ollama

# Single query
python scientific_rag_pipeline.py query "What are the main findings?"

# Interactive mode
python scientific_rag_pipeline.py interactive

# When done, free GPU memory
sudo systemctl stop ollama

📁 Repository Structure
scientific-rag-pipeline/
├── scientific_rag_pipeline.py   # Main pipeline (parsing, chunking, embedding, retrieval, generation)
├── verify_rag.py                # Chunk inspection and verification tool
├── config.yaml                  # Configuration options
├── requirements.txt             # Python dependencies
├── install_rag_pipeline.sh      # Automated installation script
├── examples.py                  # Usage examples and code snippets
├── test_installation.py         # Installation verification script
├── README.md                    # This file
└── LICENSE                      # MIT License

🔧 CLI Commands
# Ingest documents into vector store
python scientific_rag_pipeline.py ingest <path> [--extensions .pdf .txt]

# Query the knowledge base
python scientific_rag_pipeline.py query "Your question" [--top-k 8] [--stream]

# Interactive query mode
python scientific_rag_pipeline.py interactive

# Check pipeline status
python scientific_rag_pipeline.py status

# Unload LLM from GPU (free VRAM)
python scientific_rag_pipeline.py unload

Verification Tool
# List all indexed documents
python verify_rag.py list

# Inspect chunks from a document
python verify_rag.py inspect "paper_name" --num 10 --random

# Search for text across all chunks
python verify_rag.py search "Pleistocene"

# Export chunks for comparison with source PDF
python verify_rag.py export "paper_name" -o chunks.txt

# Show chunk statistics
python verify_rag.py stats

# Check document coverage
python verify_rag.py coverage "paper_name"

⚙️ Configuration
Edit config.yaml to customize:
# Storage paths (use fast NVMe)
storage:
  scratch_base: /mnt/nvme8tb
  vector_db_path: /mnt/nvme8tb/vector_store
  cache_path: /mnt/nvme8tb/cache
  document_cache: /mnt/nvme8tb/doc_cache

# Embedding model
embedding:
  model: nomic-ai/nomic-embed-text-v1.5
  batch_size: 64
  device: cuda:0

# Chunking parameters
chunking:
  chunk_size: 1024
  chunk_overlap: 128
  min_chunk_size: 100

# LLM configuration
llm:
  provider: ollama
  model: llama3.1:70b
  context_length: 32768
  base_url: http://localhost:11434

# Retrieval settings
retrieval:
  top_k: 8
  similarity_threshold: 0.35

# Hardware optimization
hardware:
  num_workers: 16
  use_gpu_parsing: true
  clear_vram_between_stages: true

🧠 VRAM Management
The pipeline uses explicit stage-based VRAM management:
Stage	Component	VRAM Usage
Ingestion	Docling (Vision models)	~3-8 GB
Ingestion	Embedding Model	~0.6 GB
Query	Embedding Model	~0.6 GB
Query	Llama 3.1 70B (Q4)	~50 GB

Important: Ollama keeps models loaded in VRAM for faster subsequent queries. To free GPU memory:
# Stop Ollama service
sudo systemctl stop ollama

# Or configure auto-unload (add to ~/.bashrc)
export OLLAMA_KEEP_ALIVE=5m

📚 Supported Document Types
·	✅ Multi-column academic PDFs
·	✅ Papers with tables and figures
·	✅ Mathematical formulas
·	✅ Mixed text/image layouts
·	✅ Scanned documents (via OCR)
🔍 Example Queries
# Factual questions
python scientific_rag_pipeline.py query "What crops were domesticated in Amazonia?"

# Temporal questions
python scientific_rag_pipeline.py query "When did humans first arrive in the Amazon?"

# Synthesis questions
python scientific_rag_pipeline.py query "What evidence exists for prehistoric agriculture?"

# Methodology questions
python scientific_rag_pipeline.py query "What methods were used to analyze phytoliths?"

🐍 Python API
from scientific_rag_pipeline import RAGConfig, ScientificRAGPipeline
from pathlib import Path

# Initialize
config = RAGConfig()
pipeline = ScientificRAGPipeline(config)

# Ingest documents
pipeline.ingest_document(Path("paper.pdf"))
pipeline.ingest_directory(Path("documents/"))

# Prepare for queries
pipeline.prepare_for_query()

# Query
result = pipeline.query("What are the main findings?")
print(f"Sources: {result['sources']}")
print(f"Answer: {result['answer']}")

# Stream response
result = pipeline.query("Explain the methodology", stream=True)
for token in result["answer_stream"]:
    print(token, end="", flush=True)

# Cleanup
pipeline.cleanup()

🔧 Troubleshooting
CUDA Out of Memory
# Check current VRAM usage
nvidia-smi

# Stop Ollama to free LLM memory
sudo systemctl stop ollama

# Use smaller model
# Edit config.yaml: model: llama3.1:8b

Ollama Connection Refused
# Start Ollama service
sudo systemctl start ollama

# Check if model is available
ollama list

# Pull model if missing
ollama pull llama3.1:70b

Slow Ingestion
# Ensure GPU parsing is enabled
# Check config.yaml: use_gpu_parsing: true

# Increase workers (up to half your CPU cores)
# Edit config.yaml: num_workers: 16

Poor Retrieval Quality
# Inspect chunks to verify content
python verify_rag.py inspect "document_name" --num 20

# Adjust chunking parameters
# Increase chunk_size for more context
# Decrease chunk_overlap if too much redundancy

📈 Model Recommendations
Embedding Models
Model	Context	Dimensions	Notes
nomic-embed-text-v1.5	8192	768	Best balance (recommended)
bge-large-en-v1.5	512	1024	Good quality, shorter context
e5-large-v2	512	1024	Alternative option

LLM Models (96GB VRAM)
Model	VRAM	Speed	Quality
llama3.1:8b	~5 GB	Fast	Good
llama3.1:70b	~50 GB	Medium	Excellent
qwen2.5:72b	~50 GB	Medium	Excellent
llama3.1:70b-q8	~75 GB	Slower	Best

🤝 Contributing
Contributions are welcome! Please feel free to submit issues and pull requests.
📄 License
MIT License - See LICENSE file for details.
🙏 Acknowledgments
·	Docling - IBM's document parsing library
·	ChromaDB - Vector database
·	Sentence Transformers - Embedding models
·	Ollama - Local LLM server
·	nomic-ai - Embedding model
