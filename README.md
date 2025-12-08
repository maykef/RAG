# Academic RAG System

Local retrieval-augmented generation for academic papers using llama.cpp, ChromaDB, and Marker PDF extraction.

## Requirements

- **Hardware**: NVIDIA GPU with sufficient VRAM (tested on RTX PRO 6000 96GB)
- **Software**: 
  - llama.cpp (built with CUDA support)
  - Mamba/Conda for environment management
  - GGUF models (embedding + LLM)

## Models

Download these models to your models directory:

1. **Embedding model**: `nomic-embed-text-v1.5.Q8_0.gguf`
   ```bash
   huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q8_0.gguf --local-dir /path/to/models
   ```

2. **LLM**: Any instruction-tuned GGUF (e.g., `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf`)

## Installation

```bash
# Run the setup script
bash rag-setup.sh

# Activate environment
mamba activate rag
```

## Configuration

Edit `rag_config.py` to set your paths:

```python
MODELS_DIR = Path("/path/to/your/models")
LLAMA_CPP_PATH = Path("/path/to/llama.cpp/build/bin")
LLM_MODEL_PATH = MODELS_DIR / "your-model.gguf"
```

## Usage

### 1. Ingest Documents

Place PDF files in the documents directory, then:

```bash
# Ingest all PDFs
python ingest.py

# Ingest single file
python ingest.py paper.pdf

# Reset database and re-ingest
python ingest.py --reset

# Test embedding generation
python ingest.py --test-embedding
```

### 2. Query

```bash
# Interactive mode (recommended)
python query.py

# Single query
python query.py "What methods were used in the study?"

# Show retrieved chunks
python query.py -v "your question"

# Retrieve more chunks
python query.py --top-k 10 "your question"
```

### Interactive Commands

| Command   | Description                      |
|-----------|----------------------------------|
| `quit`    | Exit and stop server             |
| `sources` | List ingested documents          |
| `verbose` | Toggle showing retrieved chunks  |
| `clear`   | Clear conversation history       |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDFs      │────▶│   Marker    │────▶│   Chunks    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  ChromaDB   │◀────│  Embedding  │◀────│ llama-embed │
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
       │ Query
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Retrieved  │────▶│   Prompt    │────▶│ llama-server│
│   Chunks    │     │  + Context  │     │   (LLM)     │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Files

| File              | Description                              |
|-------------------|------------------------------------------|
| `rag_config.py`   | Configuration (paths, chunk sizes, etc.) |
| `ingest.py`       | PDF ingestion and embedding              |
| `query.py`        | Interactive query interface              |
| `requirements.txt`| Python dependencies                      |

## Troubleshooting

### Embedding dimension errors
Run `python ingest.py --test-embedding` to diagnose. The system expects 768 dimensions from nomic-embed-text-v1.5.

### Server won't start
Check if port 8080 is in use: `lsof -i :8080`

### Slow ingestion
Marker PDF extraction is GPU-accelerated. Ensure CUDA is available. For very large PDFs, extraction can take several minutes.

### Zero-vector chunks
Some chunks with heavy math notation or tables may fail embedding. These are stored with zero vectors and won't be retrieved effectively. Typical failure rate is <5%.

## License

MIT
