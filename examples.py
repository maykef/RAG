#!/usr/bin/env python3
"""
Scientific RAG Pipeline - Usage Examples
=========================================

This script demonstrates how to use the Scientific RAG Pipeline
for various document processing and query tasks.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scientific_rag_pipeline import (
    RAGConfig,
    ScientificRAGPipeline,
    DoclingParser,
    MarkdownAwareChunker,
    EmbeddingEngine,
    VectorStore,
    OllamaLLM,
    vram_manager,
)


def example_basic_usage():
    """
    Example 1: Basic Usage
    ----------------------
    Simple document ingestion and querying.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60 + "\n")
    
    # Initialize with default configuration
    config = RAGConfig()
    pipeline = ScientificRAGPipeline(config)
    
    try:
        # Ingest a single document
        # pipeline.ingest_document(Path("documents/research_paper.pdf"))
        
        # Or ingest an entire directory
        # pipeline.ingest_directory(Path("documents/"), extensions=[".pdf"])
        
        # Prepare for querying (unloads parser, loads LLM)
        # pipeline.prepare_for_query()
        
        # Query the knowledge base
        # result = pipeline.query(
        #     "What are the main findings regarding protein folding?",
        #     top_k=5
        # )
        
        # print("Answer:", result["answer"])
        # print("Sources:", result["sources"])
        
        print("✓ Basic usage example complete (dry run)")
        
    finally:
        pipeline.cleanup()


def example_custom_configuration():
    """
    Example 2: Custom Configuration
    -------------------------------
    Demonstrates how to customize the pipeline for specific needs.
    """
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60 + "\n")
    
    # Create custom configuration
    config = RAGConfig(
        # Use NVMe scratch pool
        scratch_base=Path("/mnt/nvme_scratch"),
        vector_db_path=Path("/mnt/nvme_scratch/my_project_vectors"),
        
        # High-quality embedding model
        embedding_model="nomic-ai/nomic-embed-text-v1.5",
        embedding_batch_size=128,  # Larger batch for 96GB VRAM
        
        # Larger chunks for technical documents
        chunk_size=1500,
        chunk_overlap=200,
        
        # Use the largest model that fits in VRAM
        llm_model="llama3.1:70b-instruct-q8_0",  # Q8 for higher quality
        llm_context_length=65536,  # Llama 3.1 supports up to 128K
        
        # More retrieval for complex queries
        top_k=12,
        similarity_threshold=0.3,
        
        # Maximize Threadripper utilization
        num_workers=24,
    )
    
    print(f"Vector DB Path: {config.vector_db_path}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"LLM Model: {config.llm_model}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Workers: {config.num_workers}")
    print("\n✓ Custom configuration example complete")


def example_batch_processing():
    """
    Example 3: Batch Document Processing
    ------------------------------------
    Efficiently process a large corpus of documents.
    """
    print("\n" + "="*60)
    print("Example 3: Batch Document Processing")
    print("="*60 + "\n")
    
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    config = RAGConfig()
    
    # For batch processing, we handle documents in stages
    # to maximize GPU utilization
    
    def process_document(doc_path: Path) -> dict:
        """Process a single document (runs in separate process)."""
        # Each process gets its own pipeline instance
        pipeline = ScientificRAGPipeline(config)
        try:
            chunks = pipeline.ingest_document(doc_path)
            return {"path": str(doc_path), "chunks": chunks, "success": True}
        except Exception as e:
            return {"path": str(doc_path), "error": str(e), "success": False}
        finally:
            pipeline.cleanup()
    
    # Example document list
    documents = [
        # Path("documents/paper1.pdf"),
        # Path("documents/paper2.pdf"),
        # Path("documents/paper3.pdf"),
    ]
    
    # Process with multiple workers
    # Note: GPU operations are serialized, but CPU work is parallel
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     results = list(executor.map(process_document, documents))
    
    # Print results
    # for result in results:
    #     if result["success"]:
    #         print(f"✓ {result['path']}: {result['chunks']} chunks")
    #     else:
    #         print(f"✗ {result['path']}: {result['error']}")
    
    print("✓ Batch processing example complete (dry run)")


def example_streaming_response():
    """
    Example 4: Streaming LLM Response
    ---------------------------------
    Stream the LLM response for real-time output.
    """
    print("\n" + "="*60)
    print("Example 4: Streaming Response")
    print("="*60 + "\n")
    
    config = RAGConfig()
    pipeline = ScientificRAGPipeline(config)
    
    try:
        # Prepare for querying
        # pipeline.prepare_for_query()
        
        # Stream the response
        # result = pipeline.query(
        #     "Explain the methodology used in the experiments.",
        #     stream=True
        # )
        
        # print("Sources:", result["sources"])
        # print("\nAnswer:")
        # for token in result["answer_stream"]:
        #     print(token, end="", flush=True)
        # print()
        
        print("✓ Streaming example complete (dry run)")
        
    finally:
        pipeline.cleanup()


def example_filtered_retrieval():
    """
    Example 5: Filtered Retrieval
    -----------------------------
    Filter retrieval by document metadata.
    """
    print("\n" + "="*60)
    print("Example 5: Filtered Retrieval")
    print("="*60 + "\n")
    
    config = RAGConfig()
    pipeline = ScientificRAGPipeline(config)
    
    try:
        # Query with metadata filter
        # Only search within specific documents
        # result = pipeline.query(
        #     "What are the experimental results?",
        #     filter_metadata={
        #         "source": "methodology_paper.pdf"
        #     }
        # )
        
        # Or filter by section
        # result = pipeline.query(
        #     "What conclusions were drawn?",
        #     filter_metadata={
        #         "header_path": ["# Results", "## Discussion"]
        #     }
        # )
        
        print("✓ Filtered retrieval example complete (dry run)")
        
    finally:
        pipeline.cleanup()


def example_memory_optimization():
    """
    Example 6: Memory Optimization
    ------------------------------
    Demonstrates explicit VRAM management for large-scale processing.
    """
    print("\n" + "="*60)
    print("Example 6: Memory Optimization")
    print("="*60 + "\n")
    
    config = RAGConfig(
        clear_vram_between_stages=True,  # Enable stage-based cleanup
    )
    
    print("Memory management strategy for 96GB VRAM:")
    print("")
    print("Stage 1: Document Parsing")
    print("  - Docling/Marker vision models: ~8-15GB VRAM")
    print("  - Process all documents")
    print("  - Unload parser models")
    print("")
    print("Stage 2: Embedding Generation")
    print("  - nomic-embed-text: ~1-2GB VRAM")
    print("  - Process all chunks in large batches")
    print("  - Unload embedding model")
    print("")
    print("Stage 3: LLM Inference")
    print("  - Llama 3.1 70B Q4_K_M: ~40GB VRAM")
    print("  - Llama 3.1 70B Q8_0: ~70GB VRAM")
    print("  - Ready for queries")
    print("")
    
    # Log current VRAM status
    vram_manager.log_vram_status("Current")
    
    print("\n✓ Memory optimization example complete")


def example_component_usage():
    """
    Example 7: Individual Component Usage
    -------------------------------------
    Use pipeline components independently.
    """
    print("\n" + "="*60)
    print("Example 7: Individual Component Usage")
    print("="*60 + "\n")
    
    config = RAGConfig()
    
    # Use parser independently
    print("1. Document Parser:")
    parser = DoclingParser(config)
    # parser.initialize()
    # markdown = parser.parse_document(Path("document.pdf"))
    # parser.cleanup()
    print("   DoclingParser ready for use")
    
    # Use chunker independently
    print("\n2. Markdown Chunker:")
    chunker = MarkdownAwareChunker(config)
    sample_md = """
# Introduction

This is a sample document with multiple sections.

## Background

Some background information here.

## Methods

The methodology section with details.
"""
    chunks = chunker.chunk_document(sample_md, doc_id="sample")
    print(f"   Created {len(chunks)} chunks from sample Markdown")
    
    # Use embedding engine independently
    print("\n3. Embedding Engine:")
    embedder = EmbeddingEngine(config)
    # embedder.initialize()
    # embeddings = embedder.embed_texts(["Sample text 1", "Sample text 2"])
    # embedder.cleanup()
    print("   EmbeddingEngine ready for use")
    
    # Use vector store independently
    print("\n4. Vector Store:")
    # store = VectorStore(config)
    # store.initialize()
    # store.add_chunks(chunks_with_embeddings)
    # results = store.search(query_embedding, top_k=5)
    print("   VectorStore ready for use")
    
    # Use LLM independently
    print("\n5. LLM:")
    # llm = OllamaLLM(config)
    # response = llm.generate("What is the capital of France?")
    print("   OllamaLLM ready for use")
    
    print("\n✓ Component usage example complete")


def example_evaluation():
    """
    Example 8: RAG Evaluation
    -------------------------
    Evaluate retrieval quality and answer accuracy.
    """
    print("\n" + "="*60)
    print("Example 8: RAG Evaluation")
    print("="*60 + "\n")
    
    # Example evaluation queries with ground truth
    eval_dataset = [
        {
            "question": "What is the main hypothesis of the study?",
            "expected_source": "introduction.pdf",
            "expected_answer_keywords": ["hypothesis", "research"],
        },
        {
            "question": "What statistical methods were used?",
            "expected_source": "methods.pdf",
            "expected_answer_keywords": ["statistics", "analysis"],
        },
    ]
    
    print("Evaluation Metrics:")
    print("  - Retrieval Precision@K: Are relevant docs in top K?")
    print("  - Answer Relevance: Does answer address the question?")
    print("  - Faithfulness: Is answer grounded in retrieved context?")
    print("  - Source Attribution: Are sources correctly cited?")
    print("")
    
    # config = RAGConfig()
    # pipeline = ScientificRAGPipeline(config)
    # pipeline.prepare_for_query()
    
    # for eval_item in eval_dataset:
    #     result = pipeline.query(eval_item["question"])
    #     
    #     # Check retrieval quality
    #     source_found = eval_item["expected_source"] in result["sources"]
    #     
    #     # Check answer quality
    #     answer_relevant = any(
    #         kw.lower() in result["answer"].lower()
    #         for kw in eval_item["expected_answer_keywords"]
    #     )
    #     
    #     print(f"Question: {eval_item['question']}")
    #     print(f"  Source Found: {'✓' if source_found else '✗'}")
    #     print(f"  Answer Relevant: {'✓' if answer_relevant else '✗'}")
    
    print("\n✓ Evaluation example complete (dry run)")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Scientific RAG Pipeline - Usage Examples")
    print("="*60)
    
    examples = [
        example_basic_usage,
        example_custom_configuration,
        example_batch_processing,
        example_streaming_response,
        example_filtered_retrieval,
        example_memory_optimization,
        example_component_usage,
        example_evaluation,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Example failed: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
