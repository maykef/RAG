#!/usr/bin/env python3
"""
Production-Ready Local RAG Pipeline for Scientific Documents
=============================================================

Hardware Target:
    - CPU: AMD Threadripper 7970X (32-core)
    - GPU: NVIDIA RTX PRO 6000 96GB Blackwell
    - RAM: 256GB ECC RDIMM
    - Storage: 24TB NVMe Scratch Pool (ZFS)
    - OS: Ubuntu 24.04 LTS with CUDA 12.6+

Author: Claude AI Architect
License: MIT
"""

import gc
import os
import re
import sys
import json
import time
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RAG-Pipeline")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RAGConfig:
    """Central configuration for the RAG pipeline."""
    
    # Storage paths (NVMe scratch pool)
    scratch_base: Path = Path("/mnt/nvme8tb")
    vector_db_path: Path = field(default_factory=lambda: Path("/mnt/nvme8tb/vector_store"))
    cache_path: Path = field(default_factory=lambda: Path("/mnt/nvme8tb/cache"))
    document_cache: Path = field(default_factory=lambda: Path("/mnt/nvme8tb/doc_cache"))
    
    # Document processing
    input_documents_path: Path = Path("./documents")
    
    # Embedding configuration
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"  # High-quality, 8192 context
    embedding_batch_size: int = 64  # Leverage 96GB VRAM
    embedding_device: str = "cuda:0"
    
    # LLM configuration
    llm_model: str = "llama3.1:70b-instruct-q4_K_M"  # Optimal for 96GB VRAM
    llm_context_length: int = 32768
    llm_num_gpu_layers: int = -1  # All layers on GPU
    ollama_base_url: str = "http://localhost:11434"
    
    # Chunking configuration
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    
    # Retrieval configuration
    top_k: int = 8
    similarity_threshold: float = 0.35
    rerank_top_k: int = 4
    
    # Hardware optimization
    num_workers: int = 16  # Half of Threadripper cores
    use_gpu_parsing: bool = True
    clear_vram_between_stages: bool = True
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for path in [self.scratch_base, self.vector_db_path, 
                     self.cache_path, self.document_cache]:
            path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

class VRAMManager:
    """
    Explicit VRAM management for multi-stage pipeline.
    Critical for avoiding OOM even with 96GB VRAM.
    """
    
    def __init__(self):
        self._active_models: Dict[str, Any] = {}
        self._torch_available = False
        self._try_import_torch()
    
    def _try_import_torch(self):
        try:
            import torch
            self._torch = torch
            self._torch_available = True
            logger.info(f"PyTorch available: CUDA {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        except ImportError:
            logger.warning("PyTorch not available - VRAM management limited")
    
    def register_model(self, name: str, model: Any):
        """Register a model for tracking."""
        self._active_models[name] = model
        logger.debug(f"Registered model: {name}")
    
    def unload_model(self, name: str):
        """Explicitly unload a model from VRAM."""
        if name in self._active_models:
            model = self._active_models.pop(name)
            
            # Handle different model types
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                except Exception:
                    pass
            
            # Delete reference
            del model
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if self._torch_available and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
                self._torch.cuda.synchronize()
            
            logger.info(f"Unloaded model: {name}")
    
    def unload_all(self):
        """Unload all registered models."""
        for name in list(self._active_models.keys()):
            self.unload_model(name)
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage stats."""
        if not self._torch_available or not self._torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        return {
            "allocated_gb": self._torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": self._torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": self._torch.cuda.max_memory_allocated() / 1e9,
        }
    
    def log_vram_status(self, stage: str = ""):
        """Log current VRAM status."""
        stats = self.get_vram_usage()
        if "error" not in stats:
            logger.info(
                f"VRAM [{stage}] - Allocated: {stats['allocated_gb']:.2f}GB, "
                f"Reserved: {stats['reserved_gb']:.2f}GB"
            )


# Global VRAM manager instance
vram_manager = VRAMManager()


# =============================================================================
# DOCUMENT PARSING WITH DOCLING (Vision-Based)
# =============================================================================

class DoclingParser:
    """
    Vision-based document parser using IBM's Docling.
    Handles multi-column layouts, tables, and formulas via GPU acceleration.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.converter = None
        self._initialized = False
    
    def initialize(self):
        """Initialize Docling with GPU support."""
        if self._initialized:
            return
        
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            
            # Try to import accelerator options (API varies by version)
            try:
                from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions
                accelerator_options = AcceleratorOptions(
                    num_threads=self.config.num_workers,
                    device=AcceleratorDevice.CUDA if self.config.use_gpu_parsing else AcceleratorDevice.CPU,
                )
            except ImportError:
                accelerator_options = None
            
            # Try to import table structure options (API varies by version)
            try:
                from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode
                table_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
            except ImportError:
                table_options = None
            
            # Build pipeline options based on available features
            pipeline_kwargs = {
                "do_ocr": True,
                "do_table_structure": True,
            }
            
            if accelerator_options is not None:
                pipeline_kwargs["accelerator_options"] = accelerator_options
            
            if table_options is not None:
                pipeline_kwargs["table_structure_options"] = table_options
            
            pipeline_options = PdfPipelineOptions(**pipeline_kwargs)
            
            # Create format options
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
            
            # Initialize converter
            self.converter = DocumentConverter(format_options=format_options)
            self._initialized = True
            
            vram_manager.register_model("docling", self.converter)
            logger.info("Docling parser initialized with GPU acceleration")
            vram_manager.log_vram_status("Docling Init")
            
        except ImportError as e:
            logger.error(f"Docling import failed: {e}")
            logger.info("Falling back to Marker parser...")
            raise
        except Exception as e:
            logger.error(f"Docling initialization failed: {e}")
            logger.info("Falling back to Marker parser...")
            raise
    
    def parse_document(self, file_path: Path) -> str:
        """
        Parse a PDF document to clean Markdown.
        
        Returns:
            Markdown string with preserved structure.
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Parsing document: {file_path.name}")
        start_time = time.time()
        
        try:
            # Convert document
            result = self.converter.convert(str(file_path))
            
            # Export to Markdown
            markdown_content = result.document.export_to_markdown()
            
            elapsed = time.time() - start_time
            logger.info(f"Parsed {file_path.name} in {elapsed:.2f}s")
            
            return self._post_process_markdown(markdown_content)
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise
    
    def _post_process_markdown(self, content: str) -> str:
        """Clean and normalize Markdown output."""
        # Remove excessive whitespace
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        
        # Fix table formatting issues
        content = re.sub(r'\|\s*\n\s*\|', '|\n|', content)
        
        # Normalize header spacing
        content = re.sub(r'(#{1,6})\s+', r'\1 ', content)
        
        # Ensure proper spacing around headers
        content = re.sub(r'(\n)(#{1,6})', r'\n\n\2', content)
        
        return content.strip()
    
    def cleanup(self):
        """Release GPU resources."""
        if self._initialized:
            vram_manager.unload_model("docling")
            self.converter = None
            self._initialized = False
            logger.info("Docling parser cleaned up")


class MarkerParser:
    """
    Alternative vision-based parser using Marker.
    Excellent for complex scientific layouts.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """Initialize Marker with GPU support."""
        if self._initialized:
            return
        
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.config.parser import ConfigParser
            
            # Configure for GPU with high accuracy
            config_dict = {
                "torch_device": "cuda",
                "workers": self.config.num_workers,
                "force_ocr": False,  # Only OCR when needed
                "paginate_output": False,
            }
            
            parser = ConfigParser(config_dict)
            self.converter = PdfConverter(
                config=parser.generate_config(),
                artifact_dict=create_model_dict(),
            )
            
            self._initialized = True
            vram_manager.register_model("marker", self.converter)
            logger.info("Marker parser initialized with GPU acceleration")
            vram_manager.log_vram_status("Marker Init")
            
        except ImportError as e:
            logger.error(f"Marker import failed: {e}")
            raise
    
    def parse_document(self, file_path: Path) -> str:
        """Parse PDF to Markdown using Marker."""
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Parsing with Marker: {file_path.name}")
        start_time = time.time()
        
        try:
            rendered = self.converter(str(file_path))
            markdown_content = rendered.markdown
            
            elapsed = time.time() - start_time
            logger.info(f"Marker parsed {file_path.name} in {elapsed:.2f}s")
            
            return markdown_content
            
        except Exception as e:
            logger.error(f"Marker failed on {file_path}: {e}")
            raise
    
    def cleanup(self):
        """Release GPU resources."""
        if self._initialized:
            vram_manager.unload_model("marker")
            self.converter = None
            self._initialized = False
            logger.info("Marker parser cleaned up")


# =============================================================================
# MARKDOWN-AWARE CHUNKING
# =============================================================================

@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    embedding: Optional[List[float]] = None


class MarkdownAwareChunker:
    """
    Intelligent chunking that respects Markdown structure.
    Keeps headers with their content for semantic coherence.
    """
    
    # Header patterns with decreasing priority
    HEADER_PATTERNS = [
        (r'^#{1}\s+.+$', 1),   # H1
        (r'^#{2}\s+.+$', 2),   # H2
        (r'^#{3}\s+.+$', 3),   # H3
        (r'^#{4}\s+.+$', 4),   # H4
        (r'^#{5}\s+.+$', 5),   # H5
        (r'^#{6}\s+.+$', 6),   # H6
    ]
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.min_chunk_size = config.min_chunk_size
    
    def chunk_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split Markdown content into semantically coherent chunks.
        
        Strategy:
        1. First split by top-level sections (H1/H2)
        2. Then apply recursive splitting within sections
        3. Preserve header context in each chunk
        """
        metadata = metadata or {}
        chunks = []
        
        # Split into sections by H1/H2 headers
        sections = self._split_by_headers(content)
        
        for section_idx, (header_path, section_content) in enumerate(sections):
            # Add header context to each chunk from this section
            # Convert header_path list to string for ChromaDB compatibility
            section_metadata = {
                **metadata,
                "section_index": section_idx,
                "header_path": " > ".join(header_path) if header_path else "",
            }
            
            # Split section into chunks
            section_chunks = self._recursive_split(
                section_content,
                header_context=header_path[-1] if header_path else ""
            )
            
            for chunk_idx, chunk_text in enumerate(section_chunks):
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue
                
                chunk_id = hashlib.md5(
                    f"{doc_id}:{section_idx}:{chunk_idx}:{chunk_text[:50]}".encode()
                ).hexdigest()[:16]
                
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **section_metadata,
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_text),
                        "doc_id": doc_id,
                    },
                    chunk_id=chunk_id
                ))
        
        logger.info(f"Created {len(chunks)} chunks from document {doc_id}")
        return chunks
    
    def _split_by_headers(self, content: str) -> List[Tuple[List[str], str]]:
        """Split content by major headers, tracking header hierarchy."""
        lines = content.split('\n')
        sections = []
        current_headers = []
        current_content = []
        
        for line in lines:
            header_level = self._get_header_level(line)
            
            if header_level and header_level <= 2:  # Split on H1/H2
                # Save previous section
                if current_content:
                    sections.append((
                        current_headers.copy(),
                        '\n'.join(current_content)
                    ))
                
                # Update header path
                current_headers = current_headers[:header_level-1] + [line.strip()]
                current_content = [line]  # Include header in content
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections.append((current_headers.copy(), '\n'.join(current_content)))
        
        return sections
    
    def _get_header_level(self, line: str) -> Optional[int]:
        """Determine the header level of a line."""
        for pattern, level in self.HEADER_PATTERNS:
            if re.match(pattern, line.strip(), re.MULTILINE):
                return level
        return None
    
    def _recursive_split(
        self,
        text: str,
        header_context: str = ""
    ) -> List[str]:
        """
        Recursively split text respecting Markdown structure.
        Uses hierarchy of separators.
        """
        # If text is small enough, return as-is
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Separators in order of preference
        separators = [
            "\n\n\n",      # Triple newline (major break)
            "\n\n",        # Paragraph break
            "\n",          # Line break
            ". ",          # Sentence
            ", ",          # Clause
            " ",           # Word
        ]
        
        chunks = []
        for separator in separators:
            if separator in text:
                splits = text.split(separator)
                
                # Try to combine splits into appropriately sized chunks
                current_chunk = ""
                for split in splits:
                    test_chunk = (current_chunk + separator + split).strip() \
                                 if current_chunk else split
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If split itself is too large, recurse
                        if len(split) > self.chunk_size:
                            chunks.extend(self._recursive_split(split, header_context))
                        else:
                            current_chunk = split
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Apply overlap
                if self.chunk_overlap > 0:
                    chunks = self._apply_overlap(chunks)
                
                return chunks
        
        # Fallback: hard split by character
        return [text[i:i+self.chunk_size] 
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlapping context between chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0 and len(chunks[i-1]) >= self.chunk_overlap:
                # Add end of previous chunk as context
                overlap_text = chunks[i-1][-self.chunk_overlap:]
                chunk = f"...{overlap_text.strip()}\n\n{chunk}"
            overlapped.append(chunk)
        
        return overlapped


# =============================================================================
# EMBEDDING ENGINE
# =============================================================================

class EmbeddingEngine:
    """
    High-performance local embedding using sentence-transformers.
    Optimized for Blackwell GPU with large batch processing.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self._initialized = False
    
    def initialize(self):
        """Load embedding model to GPU."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            
            # Load model with trust_remote_code for nomic models
            self.model = SentenceTransformer(
                self.config.embedding_model,
                device=self.config.embedding_device,
                trust_remote_code=True,
            )
            
            # Enable torch compile for Blackwell (if supported)
            if hasattr(torch, 'compile') and torch.cuda.get_device_capability()[0] >= 8:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("Enabled torch.compile for embedding model")
                except Exception:
                    pass
            
            self._initialized = True
            vram_manager.register_model("embeddings", self.model)
            vram_manager.log_vram_status("Embeddings Init")
            
        except ImportError as e:
            logger.error(f"Failed to import sentence-transformers: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Embedding {len(texts)} texts...")
        start_time = time.time()
        
        # Add task prefix for nomic models
        if "nomic" in self.config.embedding_model.lower():
            texts = [f"search_document: {t}" for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Embedded {len(texts)} texts in {elapsed:.2f}s "
                   f"({len(texts)/elapsed:.1f} texts/sec)")
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query."""
        if not self._initialized:
            self.initialize()
        
        # Add query prefix for nomic models
        if "nomic" in self.config.embedding_model.lower():
            query = f"search_query: {query}"
        
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        
        return embedding.tolist()
    
    def cleanup(self):
        """Release GPU resources."""
        if self._initialized:
            vram_manager.unload_model("embeddings")
            self.model = None
            self._initialized = False
            logger.info("Embedding engine cleaned up")


# =============================================================================
# VECTOR STORE (ChromaDB on NVMe)
# =============================================================================

class VectorStore:
    """
    ChromaDB vector store optimized for NVMe storage.
    Supports efficient similarity search and metadata filtering.
    Uses new ChromaDB 1.x API.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._initialized = False
        self._collection_name = "scientific_docs"
    
    def initialize(self, collection_name: str = "scientific_docs"):
        """Initialize ChromaDB with persistent storage."""
        if self._initialized:
            return
        
        self._collection_name = collection_name
        
        try:
            import chromadb
            
            # New ChromaDB 1.x API - use PersistentClient for local storage
            db_path = str(self.config.vector_db_path)
            
            self.client = chromadb.PersistentClient(
                path=db_path,
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            self._initialized = True
            logger.info(f"ChromaDB initialized at {db_path}")
            logger.info(f"Collection '{collection_name}' has {self.collection.count()} vectors")
            
        except ImportError as e:
            logger.error(f"Failed to import chromadb: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks with embeddings to the vector store."""
        if not self._initialized:
            self.initialize()
        
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = [c.chunk_id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if not self._initialized:
            self.initialize()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "score": 1 - results['distances'][0][i],  # Convert distance to similarity
            })
        
        return formatted


# =============================================================================
# ALTERNATIVE: QDRANT VECTOR STORE
# =============================================================================

class QdrantVectorStore:
    """
    Qdrant vector store for production deployments.
    Offers better scalability and filtering capabilities.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = None
        self.collection_name = "scientific_docs"
        self._initialized = False
    
    def initialize(self):
        """Initialize Qdrant with local storage."""
        if self._initialized:
            return
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
            
            # Local persistent storage on NVMe
            self.client = QdrantClient(
                path=str(self.config.vector_db_path / "qdrant"),
            )
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with optimized settings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=768,  # nomic-embed-text dimension
                        distance=qmodels.Distance.COSINE,
                    ),
                    optimizers_config=qmodels.OptimizersConfigDiff(
                        indexing_threshold=20000,
                    ),
                    hnsw_config=qmodels.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            
            self._initialized = True
            logger.info("Qdrant initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import qdrant-client: {e}")
            raise
    
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to Qdrant."""
        if not self._initialized:
            self.initialize()
        
        from qdrant_client.http import models as qmodels
        
        points = []
        for chunk in chunks:
            points.append(qmodels.PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "content": chunk.content,
                    **chunk.metadata,
                }
            ))
        
        # Upsert in batches
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i+batch_size],
            )
        
        logger.info(f"Added {len(chunks)} chunks to Qdrant")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for similar chunks."""
        if not self._initialized:
            self.initialize()
        
        # Build filter if provided
        query_filter = None
        if filter_metadata:
            from qdrant_client.http import models as qmodels
            conditions = [
                qmodels.FieldCondition(
                    key=k,
                    match=qmodels.MatchValue(value=v)
                )
                for k, v in filter_metadata.items()
            ]
            query_filter = qmodels.Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )
        
        return [
            {
                "id": r.id,
                "content": r.payload.get("content", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
                "score": r.score,
            }
            for r in results
        ]


# =============================================================================
# LLM INTEGRATION (OLLAMA)
# =============================================================================

class OllamaLLM:
    """
    Ollama integration for local LLM inference.
    Optimized for Llama 3.1 70B or Qwen 2.5 72B on 96GB VRAM.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._check_model_available()
    
    def _check_model_available(self):
        """Check if the model is available in Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
            )
            
            model_name = self.config.llm_model.split(':')[0]
            if model_name not in result.stdout:
                logger.warning(f"Model {self.config.llm_model} not found locally")
                logger.info(f"Pull it with: ollama pull {self.config.llm_model}")
                
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first.")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the LLM."""
        import requests
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/chat",
                json={
                    "model": self.config.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": self.config.llm_context_length,
                        "num_gpu": self.config.llm_num_gpu_layers,
                    }
                },
                timeout=300,  # 5 minute timeout for long generations
            )
            response.raise_for_status()
            
            return response.json()["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Stream response from the LLM."""
        import requests
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.config.ollama_base_url}/api/chat",
                json={
                    "model": self.config.llm_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": self.config.llm_context_length,
                    }
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"]["content"]
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    def unload(self):
        """Explicitly unload the model from Ollama to free VRAM."""
        import requests
        
        try:
            # Setting keep_alive to 0 tells Ollama to unload immediately
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": "",
                    "keep_alive": 0,  # Unload immediately
                },
                timeout=30,
            )
            if response.status_code == 200:
                logger.info(f"Unloaded LLM model: {self.config.llm_model}")
            else:
                logger.warning(f"Could not unload model: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to unload Ollama model: {e}")


# =============================================================================
# RAG PIPELINE ORCHESTRATOR
# =============================================================================

class ScientificRAGPipeline:
    """
    Main RAG pipeline orchestrator.
    Coordinates document processing, embedding, and retrieval.
    """
    
    # Default system prompt for scientific document QA
    DEFAULT_SYSTEM_PROMPT = """You are an expert scientific research assistant with deep knowledge across multiple domains including physics, chemistry, biology, medicine, and engineering.

Your task is to answer questions based on the provided context from scientific documents. Follow these guidelines:

1. **Accuracy First**: Only use information from the provided context. If the context doesn't contain enough information, say so clearly.

2. **Technical Precision**: Use correct scientific terminology and notation. When referencing equations, tables, or figures from the context, cite them specifically.

3. **Structured Responses**: For complex questions, organize your answer with clear sections. Use bullet points for lists of findings or methods.

4. **Uncertainty Acknowledgment**: If the evidence is mixed or inconclusive, present multiple perspectives and note any limitations.

5. **Citation Practice**: Reference specific parts of the provided documents when making claims.

Context from scientific documents:
{context}

Question: {question}

Provide a comprehensive, scientifically rigorous answer:"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        # Initialize components (lazily)
        self._parser = None
        self._chunker = MarkdownAwareChunker(self.config)
        self._embedder = None
        self._vector_store = None
        self._llm = None
        
        # Tracking
        self._documents_processed = 0
    
    def _get_parser(self):
        """Get document parser (lazy initialization)."""
        if self._parser is None:
            try:
                self._parser = DoclingParser(self.config)
                self._parser.initialize()
            except ImportError:
                logger.info("Docling not available, trying Marker...")
                self._parser = MarkerParser(self.config)
                self._parser.initialize()
        return self._parser
    
    def _get_embedder(self):
        """Get embedding engine (lazy initialization)."""
        if self._embedder is None:
            self._embedder = EmbeddingEngine(self.config)
        return self._embedder
    
    def _get_vector_store(self):
        """Get vector store (lazy initialization)."""
        if self._vector_store is None:
            # Use ChromaDB by default
            self._vector_store = VectorStore(self.config)
            self._vector_store.initialize()
        return self._vector_store
    
    def _get_llm(self):
        """Get LLM (lazy initialization)."""
        if self._llm is None:
            self._llm = OllamaLLM(self.config)
        return self._llm
    
    def ingest_document(self, file_path: Path) -> int:
        """
        Ingest a single document into the RAG pipeline.
        
        Returns:
            Number of chunks created.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        logger.info(f"Ingesting document: {file_path}")
        
        # Stage 1: Parse document to Markdown
        parser = self._get_parser()
        markdown_content = parser.parse_document(file_path)
        
        # Stage 2: Chunk the Markdown
        doc_id = hashlib.md5(file_path.name.encode()).hexdigest()[:12]
        chunks = self._chunker.chunk_document(
            markdown_content,
            doc_id=doc_id,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path),
            }
        )
        
        # Stage 3: Clear parser from VRAM before embedding
        if self.config.clear_vram_between_stages:
            parser.cleanup()
            vram_manager.log_vram_status("After Parser Cleanup")
        
        # Stage 4: Generate embeddings
        embedder = self._get_embedder()
        embedder.initialize()
        
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Stage 5: Store in vector database
        vector_store = self._get_vector_store()
        vector_store.add_chunks(chunks)
        
        self._documents_processed += 1
        logger.info(f"Successfully ingested {file_path.name}: {len(chunks)} chunks")
        
        return len(chunks)
    
    def ingest_directory(self, directory: Path, extensions: List[str] = None) -> int:
        """Ingest all documents from a directory."""
        extensions = extensions or [".pdf"]
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        total_chunks = 0
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(files)} documents to ingest")
        
        for file_path in files:
            try:
                chunks = self.ingest_document(file_path)
                total_chunks += chunks
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                continue
        
        return total_chunks
    
    def prepare_for_query(self):
        """
        Prepare pipeline for query mode.
        Unloads parser and embedding models, loads LLM.
        """
        logger.info("Preparing pipeline for query mode...")
        
        # Unload processing models
        if self._parser is not None:
            self._parser.cleanup()
            self._parser = None
        
        if self._embedder is not None:
            self._embedder.cleanup()
            # Reinitialize just for query embedding (smaller memory footprint)
            self._embedder = EmbeddingEngine(self.config)
        
        vram_manager.log_vram_status("Before LLM Load")
        
        # The LLM will be loaded by Ollama on first query
        logger.info("Pipeline ready for queries")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            stream: Whether to stream the response
            
        Returns:
            Dictionary with 'answer', 'sources', and 'chunks'
        """
        top_k = top_k or self.config.top_k
        
        # Step 1: Embed the query
        embedder = self._get_embedder()
        embedder.initialize()
        query_embedding = embedder.embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        vector_store = self._get_vector_store()
        results = vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        
        # Filter by similarity threshold
        results = [r for r in results if r.get('score', 0) >= self.config.similarity_threshold]
        
        if not results:
            return {
                "answer": "I couldn't find relevant information in the documents to answer your question.",
                "sources": [],
                "chunks": [],
            }
        
        # Step 3: Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            context_parts.append(f"[Document {i}: {source}]\n{result['content']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Step 4: Generate answer with LLM
        llm = self._get_llm()
        
        prompt = self.DEFAULT_SYSTEM_PROMPT.format(
            context=context,
            question=question,
        )
        
        if stream:
            # Return generator for streaming
            def answer_stream():
                for chunk in llm.generate_stream(prompt, temperature=0.1):
                    yield chunk
            
            return {
                "answer_stream": answer_stream(),
                "sources": list(set(r['metadata'].get('source', 'Unknown') for r in results)),
                "chunks": results,
            }
        else:
            answer = llm.generate(prompt, temperature=0.1)
            
            return {
                "answer": answer,
                "sources": list(set(r['metadata'].get('source', 'Unknown') for r in results)),
                "chunks": results,
            }
    
    def cleanup(self):
        """Release all resources."""
        if self._parser:
            self._parser.cleanup()
        if self._embedder:
            self._embedder.cleanup()
        if self._llm:
            self._llm.unload()
        vram_manager.unload_all()
        logger.info("Pipeline cleanup complete")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def create_cli():
    """Create command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scientific Document RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "path",
        type=str,
        help="Path to document or directory"
    )
    ingest_parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".pdf"],
        help="File extensions to process"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument(
        "question",
        type=str,
        help="Question to ask"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to retrieve"
    )
    query_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response"
    )
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive query mode")
    
    # Status command
    subparsers.add_parser("status", help="Show pipeline status")
    
    # Unload command - free GPU memory
    subparsers.add_parser("unload", help="Unload LLM from GPU to free VRAM")
    
    return parser


def main():
    """Main entry point."""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = RAGConfig()
    pipeline = ScientificRAGPipeline(config)
    
    try:
        if args.command == "ingest":
            path = Path(args.path)
            if path.is_file():
                chunks = pipeline.ingest_document(path)
                print(f"✓ Ingested {path.name}: {chunks} chunks")
            else:
                chunks = pipeline.ingest_directory(path, args.extensions)
                print(f"✓ Ingested directory: {chunks} total chunks")
        
        elif args.command == "query":
            pipeline.prepare_for_query()
            
            if args.stream:
                result = pipeline.query(args.question, top_k=args.top_k, stream=True)
                print("\n📄 Sources:", ", ".join(result['sources']))
                print("\n💬 Answer:\n")
                for chunk in result['answer_stream']:
                    print(chunk, end="", flush=True)
                print()
            else:
                result = pipeline.query(args.question, top_k=args.top_k)
                print("\n📄 Sources:", ", ".join(result['sources']))
                print("\n💬 Answer:\n", result['answer'])
        
        elif args.command == "interactive":
            pipeline.prepare_for_query()
            print("\n🔬 Scientific RAG Pipeline - Interactive Mode")
            print("Type 'exit' or 'quit' to end session\n")
            
            while True:
                try:
                    question = input("\n❓ Your question: ").strip()
                    if question.lower() in ['exit', 'quit', 'q']:
                        break
                    if not question:
                        continue
                    
                    print("\n⏳ Searching and generating answer...")
                    result = pipeline.query(question)
                    
                    print("\n📄 Sources:", ", ".join(result['sources']))
                    print("\n💬 Answer:\n", result['answer'])
                    
                except KeyboardInterrupt:
                    break
            
            print("\n👋 Goodbye!")
        
        elif args.command == "status":
            vram_manager.log_vram_status("Current")
            vector_store = pipeline._get_vector_store()
            print(f"\n📊 Vector Store: {vector_store.collection.count()} vectors")
        
        elif args.command == "unload":
            print("🧹 Unloading LLM from GPU...")
            llm = OllamaLLM(config)
            llm.unload()
            print("✓ LLM unloaded. GPU memory should be freed.")
    
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
