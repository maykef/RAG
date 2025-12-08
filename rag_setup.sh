#!/bin/bash
# Academic RAG System Setup
# Run once to create environment and all necessary files

set -e

echo "=========================================="
echo "  Academic RAG System Setup"
echo "=========================================="

# Configuration - edit these paths as needed
MODELS_DIR="/mnt/nvme8tb/models"
RAG_WORKSPACE="$HOME/rag-workspace"
LLAMA_CPP_PATH="$HOME/llama.cpp/build/bin"
LLM_MODEL="Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
EMBEDDING_MODEL="nomic-embed-text-v1.5.Q8_0.gguf"

# Create directories
echo "[1/6] Creating directories..."
mkdir -p "$RAG_WORKSPACE"
mkdir -p "$MODELS_DIR/rag-db"
mkdir -p "$MODELS_DIR/rag-documents"

# Create mamba environment
echo "[2/6] Creating mamba environment..."
if mamba env list | grep -q "^rag "; then
    echo "Environment 'rag' already exists, skipping creation"
else
    mamba create -n rag python=3.11 -y
fi

# Install dependencies
echo "[3/6] Installing Python dependencies..."
mamba run -n rag pip install --break-system-packages \
    chromadb>=0.4.0 \
    langchain-text-splitters>=0.0.1 \
    requests>=2.28.0 \
    rich>=13.0.0 \
    tqdm>=4.65.0 \
    marker-pdf>=0.1.0 \
    pymupdf>=1.23.0

# Download embedding model if not present
echo "[4/6] Checking embedding model..."
if [ ! -f "$MODELS_DIR/$EMBEDDING_MODEL" ]; then
    echo "Downloading embedding model..."
    mamba run -n rag huggingface-cli download \
        nomic-ai/nomic-embed-text-v1.5-GGUF \
        "$EMBEDDING_MODEL" \
        --local-dir "$MODELS_DIR"
else
    echo "Embedding model already exists"
fi

# Create Python files
echo "[5/6] Creating Python scripts..."

# rag_config.py
cat > "$RAG_WORKSPACE/rag_config.py" << 'PYEOF'
"""RAG Configuration - Edit paths here"""

from pathlib import Path

# Paths
MODELS_DIR = Path("/mnt/nvme8tb/models")
RAG_DB_DIR = MODELS_DIR / "rag-db"
DOCUMENTS_DIR = MODELS_DIR / "rag-documents"

# Embedding model (GGUF for llama.cpp)
EMBEDDING_MODEL_PATH = MODELS_DIR / "nomic-embed-text-v1.5.Q8_0.gguf"

# LLM settings
LLAMA_CPP_PATH = Path.home() / "llama.cpp/build/bin"
LLM_MODEL_PATH = MODELS_DIR / "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"

# Chunking settings
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 100

# Retrieval settings
TOP_K = 5  # Number of chunks to retrieve

# ChromaDB collection name
COLLECTION_NAME = "academic_papers"
PYEOF

# ingest.py
cat > "$RAG_WORKSPACE/ingest.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Ingest academic PDFs into the RAG database.

Usage:
  python ingest.py                     # Ingest all PDFs in documents folder
  python ingest.py paper.pdf           # Ingest single file
  python ingest.py /path/to/folder     # Ingest all PDFs in folder
  python ingest.py --reset             # Clear database and re-ingest all
  python ingest.py --test-embedding    # Test embedding generation
"""

import argparse
import subprocess
import tempfile
import hashlib
import re
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_config import (
    RAG_DB_DIR, DOCUMENTS_DIR, EMBEDDING_MODEL_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, LLAMA_CPP_PATH
)

console = Console()

# Expected embedding dimension for nomic-embed-text-v1.5
EMBEDDING_DIM = 768


class Embedder:
    """Generate embeddings using llama.cpp"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.llama_embedding = LLAMA_CPP_PATH / "llama-embedding"
        if not self.llama_embedding.exists():
            raise FileNotFoundError(f"llama-embedding not found at {self.llama_embedding}")
        
        # Test embedding on init and detect actual dimension
        console.print("[blue]Testing embedding model...[/blue]")
        test_emb = self._embed_single_raw("test embedding dimension check")
        self.actual_dim = len(test_emb)
        console.print(f"[green]Model outputs {self.actual_dim} dimensions[/green]")
        
        if self.actual_dim < EMBEDDING_DIM:
            raise ValueError(f"Model dimension {self.actual_dim} < expected {EMBEDDING_DIM}")
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with consistent dimensions"""
        embeddings = []
        failed_count = 0
        
        for i, text in enumerate(tqdm(texts, desc="Generating embeddings", leave=False)):
            emb = self._embed_single(text)
            
            if emb is None:
                console.print(f"[yellow]Chunk {i}: embedding failed, using zero vector[/yellow]")
                emb = [0.0] * EMBEDDING_DIM
                failed_count += 1
            
            embeddings.append(emb)
        
        if failed_count > 0:
            console.print(f"[yellow]Warning: {failed_count} chunks used fallback zero embeddings[/yellow]")
        
        return embeddings
    
    def _embed_single(self, text: str) -> list[float] | None:
        """Embed single text, return exactly EMBEDDING_DIM floats"""
        text_clean = self._clean_text(text)
        
        if not text_clean.strip():
            return [0.0] * EMBEDDING_DIM
        
        raw_emb = self._embed_single_raw(text_clean)
        
        if len(raw_emb) >= EMBEDDING_DIM:
            return raw_emb[:EMBEDDING_DIM]
        
        # Retry with aggressive cleaning
        raw_emb = self._embed_single_raw(self._aggressive_clean(text))
        
        if len(raw_emb) >= EMBEDDING_DIM:
            return raw_emb[:EMBEDDING_DIM]
        
        return None
    
    def _embed_single_raw(self, text: str) -> list[float]:
        """Get raw embedding output as list of floats"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            tmp_path = f.name
        
        try:
            result = subprocess.run(
                [
                    str(self.llama_embedding),
                    "-m", str(self.model_path),
                    "-f", tmp_path,
                    "--pooling", "mean",
                    "-ngl", "999"
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            embedding = self._parse_floats(result.stdout)
            
            if len(embedding) < EMBEDDING_DIM:
                embedding = self._parse_floats(result.stderr)
            
            return embedding
            
        except subprocess.TimeoutExpired:
            console.print(f"[red]Embedding timeout[/red]")
            return []
        except Exception as e:
            console.print(f"[red]Embedding error: {e}[/red]")
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _parse_floats(self, output: str) -> list[float]:
        """Parse floats from output, filtering out non-embedding values"""
        embedding = []
        float_pattern = re.compile(r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+')
        
        for match in float_pattern.finditer(output):
            try:
                val = float(match.group())
                if abs(val) < 100:
                    embedding.append(val)
            except ValueError:
                continue
        
        return embedding
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding"""
        text = text[:6000]
        text = text.replace('\x00', ' ')
        text = ' '.join(text.split())
        text = ''.join(c if c.isprintable() or c in '\n\t ' else ' ' for c in text)
        return text
    
    def _aggressive_clean(self, text: str) -> str:
        """Aggressively clean text - ASCII only"""
        text = ''.join(c if ord(c) < 128 and c.isprintable() else ' ' for c in text)
        text = ' '.join(text.split())
        return text[:4000]


def extract_text_marker(pdf_path: Path) -> str:
    """Extract text from PDF using Marker"""
    console.print(f"[blue]Extracting text from {pdf_path.name}...[/blue]")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = subprocess.run(
                [
                    "marker_single", str(pdf_path),
                    "--output_dir", tmpdir,
                    "--output_format", "markdown",
                    "--disable_image_extraction"
                ],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            md_files = list(Path(tmpdir).rglob("*.md"))
            if md_files:
                text = md_files[0].read_text(encoding='utf-8', errors='replace')
                console.print(f"[green]Marker extraction successful ({len(text)} chars)[/green]")
                return text
            else:
                console.print(f"[yellow]Marker produced no output, falling back to PyMuPDF[/yellow]")
                return extract_text_pymupdf(pdf_path)
                
        except subprocess.TimeoutExpired:
            console.print(f"[yellow]Marker timeout, falling back to PyMuPDF[/yellow]")
            return extract_text_pymupdf(pdf_path)
        except FileNotFoundError:
            console.print(f"[yellow]Marker not found, using PyMuPDF[/yellow]")
            return extract_text_pymupdf(pdf_path)
        except Exception as e:
            console.print(f"[yellow]Marker error: {e}, falling back to PyMuPDF[/yellow]")
            return extract_text_pymupdf(pdf_path)


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Fallback text extraction using PyMuPDF"""
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file for deduplication"""
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def ingest_document(pdf_path: Path, collection, embedder, splitter) -> int:
    """Ingest a single document, return number of chunks added"""
    
    file_hash = get_file_hash(pdf_path)
    
    # Check if already ingested
    existing = collection.get(where={"file_hash": file_hash})
    if existing and len(existing['ids']) > 0:
        console.print(f"[yellow]Skipping {pdf_path.name} (already ingested)[/yellow]")
        return 0
    
    # Extract text
    text = extract_text_marker(pdf_path)
    if not text.strip():
        console.print(f"[red]No text extracted from {pdf_path.name}[/red]")
        return 0
    
    # Chunk
    chunks = splitter.split_text(text)
    console.print(f"[blue]Split into {len(chunks)} chunks[/blue]")
    
    if len(chunks) == 0:
        console.print(f"[red]No chunks created from {pdf_path.name}[/red]")
        return 0
    
    # Generate embeddings
    embeddings = embedder.embed(chunks)
    
    # Store in ChromaDB
    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": pdf_path.name,
            "file_hash": file_hash,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return len(chunks)


def test_embedding(embedder):
    """Test embedding generation"""
    console.print("[bold]Testing embedding generation...[/bold]\n")
    
    test_cases = [
        "Simple English text.",
        "The mitochondria is the powerhouse of the cell.",
        "∫ f(x)dx = F(x) + C",
        "Röntgen discovered X-rays in 1895.",
        "DNA序列分析",
        "a " * 2000,
    ]
    
    all_good = True
    for i, text in enumerate(test_cases):
        emb = embedder._embed_single(text)
        dim = len(emb) if emb else 0
        status = "✓" if dim == EMBEDDING_DIM else "✗"
        if dim != EMBEDDING_DIM:
            all_good = False
        console.print(f"{status} Test {i+1}: {dim} dims | {text[:50]}...")
    
    console.print(f"\n[{'green' if all_good else 'red'}]Expected dimension: {EMBEDDING_DIM}[/]")
    
    if all_good:
        console.print("[green]All tests passed![/green]")
    else:
        console.print("[red]Some tests failed - check llama-embedding output[/red]")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into RAG database")
    parser.add_argument("path", nargs="?", default=None, help="PDF file or folder")
    parser.add_argument("--reset", action="store_true", help="Reset database before ingesting")
    parser.add_argument("--test-embedding", action="store_true", help="Test embedding generation")
    args = parser.parse_args()
    
    console.print("[bold green]=== Academic RAG Ingestion ===[/bold green]\n")
    
    # Initialize embedder
    embedder = Embedder(EMBEDDING_MODEL_PATH)
    
    if args.test_embedding:
        test_embedding(embedder)
        return
    
    # Determine input path
    if args.path:
        input_path = Path(args.path)
    else:
        input_path = DOCUMENTS_DIR
    
    # Collect PDFs
    if input_path.is_file():
        pdfs = [input_path]
    else:
        pdfs = list(input_path.glob("*.pdf"))
    
    if not pdfs:
        console.print(f"[red]No PDFs found in {input_path}[/red]")
        console.print(f"[yellow]Place PDF files in: {DOCUMENTS_DIR}[/yellow]")
        return
    
    console.print(f"Found {len(pdfs)} PDF(s) to process\n")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path=str(RAG_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            console.print("[yellow]Database reset[/yellow]")
        except:
            pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Academic papers for RAG"}
    )
    
    # Initialize splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Process each PDF
    total_chunks = 0
    successful = 0
    failed = 0
    
    for pdf in tqdm(pdfs, desc="Processing PDFs"):
        try:
            chunks_added = ingest_document(pdf, collection, embedder, splitter)
            total_chunks += chunks_added
            if chunks_added > 0:
                console.print(f"[green]✓ {pdf.name}: {chunks_added} chunks[/green]")
                successful += 1
            else:
                failed += 1
        except Exception as e:
            console.print(f"[red]✗ {pdf.name}: {e}[/red]")
            failed += 1
    
    # Summary
    console.print(f"\n[bold green]=== Ingestion Complete ===[/bold green]")
    console.print(f"Successful: {successful} | Failed: {failed}")
    console.print(f"Total chunks added: {total_chunks}")
    console.print(f"Total documents in DB: {collection.count()}")


if __name__ == "__main__":
    main()
PYEOF

# query.py
cat > "$RAG_WORKSPACE/query.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Academic RAG Query System

Automatically starts llama-server, runs interactive queries with conversation memory,
and cleans up on exit.

Usage:
  python query.py                      # Interactive mode
  python query.py "your question"      # Single query then exit
  python query.py --top-k 10           # Retrieve more chunks
"""

import argparse
import atexit
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from collections import deque

import requests
import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live

from rag_config import (
    RAG_DB_DIR, EMBEDDING_MODEL_PATH, COLLECTION_NAME,
    LLAMA_CPP_PATH, LLM_MODEL_PATH, TOP_K
)

console = Console()

LLAMA_SERVER_URL = "http://127.0.0.1:8080"
HISTORY_SIZE = 5  # Number of Q&A pairs to keep
server_process = None


def start_server():
    """Start llama-server in background"""
    global server_process
    
    if check_server():
        console.print("[green]✓ llama-server already running[/green]")
        return True
    
    console.print("[yellow]Starting llama-server...[/yellow]")
    
    llama_server = LLAMA_CPP_PATH / "llama-server"
    
    server_process = subprocess.Popen(
        [
            str(llama_server),
            "-m", str(LLM_MODEL_PATH),
            "-ngl", "999",
            "-c", "8192",
            "--host", "127.0.0.1",
            "--port", "8080"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
    )
    
    with Live(Spinner("dots", text="Loading model..."), console=console, transient=True):
        for i in range(120):
            time.sleep(1)
            if check_server():
                console.print("[green]✓ llama-server ready[/green]")
                return True
            if server_process.poll() is not None:
                console.print("[red]✗ Server failed to start[/red]")
                return False
    
    console.print("[red]✗ Server startup timeout[/red]")
    return False


def stop_server():
    """Stop llama-server if we started it"""
    global server_process
    if server_process is not None:
        console.print("\n[yellow]Stopping llama-server...[/yellow]")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
        server_process = None
        console.print("[green]✓ Server stopped[/green]")


def check_server():
    """Check if llama-server is responding"""
    try:
        r = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


class Embedder:
    """Generate query embeddings using llama.cpp"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.llama_embedding = LLAMA_CPP_PATH / "llama-embedding"
        
    def embed(self, text: str) -> list[float]:
        """Embed query text"""
        text_clean = ' '.join(text.split())[:4000]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text_clean)
            tmp_path = f.name
        
        try:
            result = subprocess.run(
                [
                    str(self.llama_embedding),
                    "-m", str(self.model_path),
                    "-f", tmp_path,
                    "--pooling", "mean",
                    "-ngl", "999"
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            float_pattern = re.compile(r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+')
            embedding = []
            for match in float_pattern.finditer(result.stdout):
                val = float(match.group())
                if abs(val) < 100:
                    embedding.append(val)
            
            return embedding[:768]
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class ConversationHistory:
    """Maintains rolling conversation history"""
    
    def __init__(self, max_size: int = HISTORY_SIZE):
        self.history = deque(maxlen=max_size)
    
    def add(self, question: str, answer: str):
        """Add a Q&A pair"""
        self.history.append({"q": question, "a": answer})
    
    def clear(self):
        """Clear history"""
        self.history.clear()
    
    def format_for_prompt(self) -> str:
        """Format history for inclusion in prompt"""
        if not self.history:
            return ""
        
        parts = ["PREVIOUS CONVERSATION:"]
        for exchange in self.history:
            parts.append(f"User: {exchange['q']}")
            answer = exchange['a']
            if len(answer) > 500:
                answer = answer[:500] + "..."
            parts.append(f"Assistant: {answer}")
        parts.append("")
        
        return "\n".join(parts)


def retrieve(query: str, collection, embedder, top_k: int) -> list[dict]:
    """Retrieve relevant chunks for a query"""
    
    query_embedding = embedder.embed(query)
    
    if len(query_embedding) != 768:
        console.print(f"[red]Query embedding failed[/red]")
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    for i in range(len(results['ids'][0])):
        chunks.append({
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'chunk_index': results['metadatas'][0][i]['chunk_index'],
            'distance': results['distances'][0][i]
        })
    
    return chunks


def build_prompt(query: str, chunks: list[dict], history: ConversationHistory) -> str:
    """Build prompt with conversation history and retrieved context"""
    
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    history_text = history.format_for_prompt()
    
    return f"""You are a research assistant. Answer based on the provided context from academic papers.
Cite sources when making claims. If context is insufficient, say so.
Use the conversation history to understand follow-up questions and references like "it", "that paper", "the same", etc.

{history_text}CONTEXT FROM DOCUMENTS:
{context}

CURRENT QUESTION: {query}

ANSWER:"""


def query_llm(prompt: str) -> str:
    """Query the LLM via HTTP API"""
    
    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": 1024,
                "temperature": 0.3,
                "stop": ["CURRENT QUESTION:", "\n\nCURRENT QUESTION", "User:"],
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("content", "").strip()
        else:
            return f"[Error: Server returned {response.status_code}]"
            
    except requests.exceptions.Timeout:
        return "[Error: Request timed out]"
    except Exception as e:
        return f"[Error: {e}]"


def show_sources(collection):
    """Show all source documents"""
    results = collection.get(include=['metadatas'])
    sources = {}
    for meta in results['metadatas']:
        src = meta['source']
        sources[src] = sources.get(src, 0) + 1
    
    console.print("\n[bold]Documents in database:[/bold]")
    for src, count in sorted(sources.items()):
        console.print(f"  • {src}: {count} chunks")


def process_query(query: str, collection, embedder, top_k: int, 
                  history: ConversationHistory, verbose: bool = False) -> str:
    """Process a single query, return the answer"""
    
    with Live(Spinner("dots", text="Searching..."), console=console, transient=True):
        chunks = retrieve(query, collection, embedder, top_k)
    
    if not chunks:
        console.print("[red]No relevant chunks found[/red]")
        return ""
    
    sources = set(c['source'] for c in chunks)
    console.print(f"[dim]Sources: {', '.join(sources)}[/dim]")
    
    if verbose:
        for i, chunk in enumerate(chunks, 1):
            console.print(f"\n[dim]── Chunk {i} (dist: {chunk['distance']:.3f}) ──[/dim]")
            console.print(chunk['text'][:400] + "..." if len(chunk['text']) > 400 else chunk['text'])
        console.print()
    
    with Live(Spinner("dots", text="Generating answer..."), console=console, transient=True):
        prompt = build_prompt(query, chunks, history)
        answer = query_llm(prompt)
    
    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
    
    return answer


def interactive_mode(collection, embedder, top_k: int, verbose: bool):
    """Run interactive query loop with conversation memory"""
    
    history = ConversationHistory()
    
    console.print(Panel.fit(
        f"[bold green]Academic RAG[/bold green]\n"
        f"Documents: {collection.count()} chunks\n"
        f"Memory: last {HISTORY_SIZE} exchanges\n"
        f"Commands: [dim]quit, sources, verbose, clear[/dim]",
        title="Ready"
    ))
    
    while True:
        try:
            query = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not query:
            continue
        
        cmd = query.lower()
        if cmd in ('quit', 'exit', 'q'):
            break
        if cmd == 'sources':
            show_sources(collection)
            continue
        if cmd == 'verbose':
            verbose = not verbose
            console.print(f"[yellow]Verbose mode: {'on' if verbose else 'off'}[/yellow]")
            continue
        if cmd == 'clear':
            history.clear()
            console.print("[yellow]Conversation history cleared[/yellow]")
            continue
        
        answer = process_query(query, collection, embedder, top_k, history, verbose)
        
        if answer and not answer.startswith("[Error"):
            history.add(query, answer)


def main():
    parser = argparse.ArgumentParser(description="Academic RAG Query System")
    parser.add_argument("query", nargs="?", default=None, help="Single query (omit for interactive mode)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show retrieved chunks")
    args = parser.parse_args()
    
    # Register cleanup
    atexit.register(stop_server)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    
    # Start server
    if not start_server():
        sys.exit(1)
    
    # Initialize database
    client = chromadb.PersistentClient(
        path=str(RAG_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        console.print("[red]Database not found. Run ingest.py first.[/red]")
        sys.exit(1)
    
    if collection.count() == 0:
        console.print("[red]Database empty. Run ingest.py first.[/red]")
        sys.exit(1)
    
    embedder = Embedder(EMBEDDING_MODEL_PATH)
    
    if args.query:
        history = ConversationHistory()
        process_query(args.query, collection, embedder, args.top_k, history, args.verbose)
    else:
        interactive_mode(collection, embedder, args.top_k, args.verbose)


if __name__ == "__main__":
    main()
PYEOF

# requirements.txt
cat > "$RAG_WORKSPACE/requirements.txt" << 'EOF'
chromadb>=0.4.0
langchain-text-splitters>=0.0.1
requests>=2.28.0
rich>=13.0.0
tqdm>=4.65.0
marker-pdf>=0.1.0
pymupdf>=1.23.0
EOF

# Make scripts executable
chmod +x "$RAG_WORKSPACE/ingest.py"
chmod +x "$RAG_WORKSPACE/query.py"

# Create README
echo "[6/6] Creating documentation..."
cat > "$RAG_WORKSPACE/README.md" << 'EOF'
# Academic RAG System

Local retrieval-augmented generation for academic papers using llama.cpp, ChromaDB, and Marker PDF extraction.

## Requirements

- **Hardware**: NVIDIA GPU with sufficient VRAM
- **Software**: llama.cpp (built with CUDA), Mamba/Conda

## Quick Start

```bash
# Activate environment
mamba activate rag

# Add PDFs to the documents folder
cp your-papers/*.pdf /mnt/nvme8tb/models/rag-documents/

# Ingest documents
python ingest.py

# Query (starts server automatically)
python query.py
```

## Usage

### Ingestion

```bash
python ingest.py                     # Ingest all PDFs
python ingest.py paper.pdf           # Single file
python ingest.py --reset             # Clear and re-ingest
python ingest.py --test-embedding    # Test embeddings
```

### Query

```bash
python query.py                      # Interactive mode
python query.py "your question"      # Single query
python query.py -v "question"        # Show retrieved chunks
python query.py --top-k 10           # More chunks
```

### Interactive Commands

- `quit` - Exit and stop server
- `sources` - List documents
- `verbose` - Toggle chunk display
- `clear` - Clear conversation history

## Configuration

Edit `rag_config.py` to change paths, chunk sizes, or model settings.

## Architecture

- **PDF Extraction**: Marker (GPU-accelerated)
- **Embeddings**: nomic-embed-text-v1.5 via llama-embedding
- **Vector DB**: ChromaDB
- **LLM**: llama-server (auto-managed)
- **Conversation**: Rolling 5-exchange history
EOF

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:  mamba activate rag"
echo "  2. Add PDFs to:           $MODELS_DIR/rag-documents/"
echo "  3. Ingest documents:      cd $RAG_WORKSPACE && python ingest.py"
echo "  4. Start querying:        python query.py"
echo ""
