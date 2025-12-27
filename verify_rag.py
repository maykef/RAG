#!/usr/bin/env python3
"""
RAG Verification Script
=======================
Tools to inspect and verify chunk quality against source documents.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional
import chromadb


def get_collection(db_path: str = "/mnt/nvme8tb/vector_store", collection_name: str = "scientific_docs"):
    """Connect to ChromaDB and get collection."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(collection_name)


def list_documents(collection):
    """List all unique documents in the collection."""
    # Get all metadata
    results = collection.get(include=["metadatas"])
    
    # Extract unique documents
    docs = {}
    for meta in results["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in docs:
            docs[source] = {"chunks": 0, "doc_id": meta.get("doc_id", "")}
        docs[source]["chunks"] += 1
    
    print(f"\n{'='*60}")
    print(f" Documents in Collection: {len(docs)}")
    print(f" Total Chunks: {len(results['metadatas'])}")
    print(f"{'='*60}\n")
    
    for i, (source, info) in enumerate(sorted(docs.items()), 1):
        print(f"  {i:2}. {Path(source).name}")
        print(f"      Chunks: {info['chunks']}, Doc ID: {info['doc_id']}")
    
    return docs


def inspect_document(collection, source_pattern: str, num_chunks: int = 5, random_sample: bool = False):
    """Inspect chunks from a specific document."""
    # Get all chunks
    results = collection.get(include=["documents", "metadatas"])
    
    # Filter by source pattern
    matching = []
    for i, meta in enumerate(results["metadatas"]):
        source = meta.get("source", "")
        if source_pattern.lower() in source.lower():
            matching.append({
                "content": results["documents"][i],
                "metadata": meta,
                "id": results["ids"][i]
            })
    
    if not matching:
        print(f"No chunks found matching '{source_pattern}'")
        return
    
    print(f"\n{'='*60}")
    print(f" Found {len(matching)} chunks matching '{source_pattern}'")
    print(f"{'='*60}")
    
    # Select chunks to display
    if random_sample:
        selected = random.sample(matching, min(num_chunks, len(matching)))
        print(f" Showing {len(selected)} RANDOM chunks:\n")
    else:
        selected = matching[:num_chunks]
        print(f" Showing first {len(selected)} chunks:\n")
    
    for i, chunk in enumerate(selected, 1):
        meta = chunk["metadata"]
        print(f"{'─'*60}")
        print(f"CHUNK {i} | ID: {chunk['id']}")
        print(f"Section: {meta.get('section_index', '?')} | "
              f"Chunk Index: {meta.get('chunk_index', '?')} | "
              f"Chars: {meta.get('char_count', '?')}")
        print(f"Header Path: {meta.get('header_path', 'N/A')}")
        print(f"{'─'*60}")
        print(chunk["content"][:1500])
        if len(chunk["content"]) > 1500:
            print(f"\n... [truncated, {len(chunk['content'])} total chars]")
        print()


def search_content(collection, query: str, num_results: int = 5):
    """Search for specific text in chunks (exact match, not semantic)."""
    results = collection.get(include=["documents", "metadatas"])
    
    matching = []
    query_lower = query.lower()
    
    for i, doc in enumerate(results["documents"]):
        if query_lower in doc.lower():
            matching.append({
                "content": doc,
                "metadata": results["metadatas"][i],
                "id": results["ids"][i]
            })
    
    print(f"\n{'='*60}")
    print(f" Text Search: '{query}'")
    print(f" Found {len(matching)} chunks containing this text")
    print(f"{'='*60}\n")
    
    for i, chunk in enumerate(matching[:num_results], 1):
        meta = chunk["metadata"]
        source = Path(meta.get("source", "unknown")).name
        
        # Highlight the match
        content = chunk["content"]
        idx = content.lower().find(query_lower)
        start = max(0, idx - 100)
        end = min(len(content), idx + len(query) + 100)
        snippet = content[start:end]
        
        print(f"{'─'*60}")
        print(f"MATCH {i} | Source: {source}")
        print(f"Section: {meta.get('section_index', '?')} | Header: {meta.get('header_path', 'N/A')[:50]}")
        print(f"{'─'*60}")
        print(f"...{snippet}...")
        print()


def export_chunks(collection, source_pattern: str, output_file: str):
    """Export all chunks from a document to a text file for comparison."""
    results = collection.get(include=["documents", "metadatas"])
    
    # Filter and sort by section/chunk index
    matching = []
    for i, meta in enumerate(results["metadatas"]):
        source = meta.get("source", "")
        if source_pattern.lower() in source.lower():
            matching.append({
                "content": results["documents"][i],
                "metadata": meta,
                "section_index": meta.get("section_index", 0),
                "chunk_index": meta.get("chunk_index", 0)
            })
    
    if not matching:
        print(f"No chunks found matching '{source_pattern}'")
        return
    
    # Sort by section then chunk index
    matching.sort(key=lambda x: (x["section_index"], x["chunk_index"]))
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(f"# Exported chunks for: {source_pattern}\n")
        f.write(f"# Total chunks: {len(matching)}\n")
        f.write(f"# Use this to compare against the original PDF\n\n")
        
        for chunk in matching:
            meta = chunk["metadata"]
            f.write(f"{'='*80}\n")
            f.write(f"SECTION {meta.get('section_index', '?')} | CHUNK {meta.get('chunk_index', '?')}\n")
            f.write(f"Header: {meta.get('header_path', 'N/A')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(chunk["content"])
            f.write("\n\n")
    
    print(f"✓ Exported {len(matching)} chunks to: {output_file}")


def chunk_statistics(collection):
    """Show statistics about chunk sizes and distribution."""
    results = collection.get(include=["documents", "metadatas"])
    
    # Gather stats
    char_counts = []
    docs_chunks = {}
    
    for i, meta in enumerate(results["metadatas"]):
        char_count = len(results["documents"][i])
        char_counts.append(char_count)
        
        source = Path(meta.get("source", "unknown")).name
        if source not in docs_chunks:
            docs_chunks[source] = []
        docs_chunks[source].append(char_count)
    
    print(f"\n{'='*60}")
    print(f" Chunk Statistics")
    print(f"{'='*60}\n")
    
    print(f"Total chunks: {len(char_counts)}")
    print(f"Total characters: {sum(char_counts):,}")
    print(f"\nChunk size distribution:")
    print(f"  Min:    {min(char_counts):,} chars")
    print(f"  Max:    {max(char_counts):,} chars")
    print(f"  Mean:   {sum(char_counts)//len(char_counts):,} chars")
    
    # Size buckets
    buckets = {"<256": 0, "256-512": 0, "512-1024": 0, "1024-2048": 0, ">2048": 0}
    for c in char_counts:
        if c < 256:
            buckets["<256"] += 1
        elif c < 512:
            buckets["256-512"] += 1
        elif c < 1024:
            buckets["512-1024"] += 1
        elif c < 2048:
            buckets["1024-2048"] += 1
        else:
            buckets[">2048"] += 1
    
    print(f"\nSize distribution:")
    for bucket, count in buckets.items():
        pct = count * 100 // len(char_counts)
        bar = "█" * (pct // 2)
        print(f"  {bucket:>10}: {count:4} ({pct:2}%) {bar}")
    
    print(f"\nPer-document breakdown:")
    for doc, chunks in sorted(docs_chunks.items()):
        avg = sum(chunks) // len(chunks)
        print(f"  {doc[:40]:40} | {len(chunks):3} chunks | avg {avg:,} chars")


def verify_coverage(collection, source_pattern: str, pdf_text_file: Optional[str] = None):
    """
    Check if chunks cover the full document.
    Optionally compare against extracted PDF text.
    """
    results = collection.get(include=["documents", "metadatas"])
    
    # Get chunks for this document
    chunks = []
    for i, meta in enumerate(results["metadatas"]):
        source = meta.get("source", "")
        if source_pattern.lower() in source.lower():
            chunks.append({
                "content": results["documents"][i],
                "section": meta.get("section_index", 0),
                "chunk": meta.get("chunk_index", 0)
            })
    
    if not chunks:
        print(f"No chunks found matching '{source_pattern}'")
        return
    
    # Sort and concatenate
    chunks.sort(key=lambda x: (x["section"], x["chunk"]))
    full_text = "\n\n".join(c["content"] for c in chunks)
    
    print(f"\n{'='*60}")
    print(f" Coverage Verification: {source_pattern}")
    print(f"{'='*60}\n")
    print(f"Chunks found: {len(chunks)}")
    print(f"Total reconstructed text: {len(full_text):,} characters")
    
    if pdf_text_file:
        with open(pdf_text_file) as f:
            original = f.read()
        
        print(f"Original text file: {len(original):,} characters")
        
        # Simple coverage check - what % of original words are in chunks
        original_words = set(original.lower().split())
        chunk_words = set(full_text.lower().split())
        
        covered = len(original_words & chunk_words)
        coverage = covered * 100 / len(original_words) if original_words else 0
        
        print(f"\nWord coverage: {coverage:.1f}%")
        print(f"  Original unique words: {len(original_words):,}")
        print(f"  Chunk unique words: {len(chunk_words):,}")
        print(f"  Overlapping words: {covered:,}")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Verification Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all documents in the collection
  python verify_rag.py list
  
  # Inspect chunks from a specific paper
  python verify_rag.py inspect "Iriarte" --num 10
  
  # Random sample of chunks from a paper
  python verify_rag.py inspect "Iriarte" --num 5 --random
  
  # Search for specific text across all chunks
  python verify_rag.py search "climate change"
  
  # Export all chunks from a paper to a file
  python verify_rag.py export "Iriarte" -o iriarte_chunks.txt
  
  # Show chunk statistics
  python verify_rag.py stats
  
  # Check document coverage
  python verify_rag.py coverage "Iriarte"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all documents")
    
    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect chunks from a document")
    inspect_parser.add_argument("pattern", help="Document name pattern to match")
    inspect_parser.add_argument("--num", "-n", type=int, default=5, help="Number of chunks to show")
    inspect_parser.add_argument("--random", "-r", action="store_true", help="Random sample")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for text in chunks")
    search_parser.add_argument("query", help="Text to search for")
    search_parser.add_argument("--num", "-n", type=int, default=5, help="Number of results")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export chunks to file")
    export_parser.add_argument("pattern", help="Document name pattern")
    export_parser.add_argument("--output", "-o", required=True, help="Output file")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show chunk statistics")
    
    # Coverage command
    coverage_parser = subparsers.add_parser("coverage", help="Verify document coverage")
    coverage_parser.add_argument("pattern", help="Document name pattern")
    coverage_parser.add_argument("--compare", "-c", help="Original text file to compare")
    
    # Global options
    parser.add_argument("--db", default="/mnt/nvme8tb/vector_store", help="ChromaDB path")
    parser.add_argument("--collection", default="scientific_docs", help="Collection name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Connect to collection
    collection = get_collection(args.db, args.collection)
    
    # Run command
    if args.command == "list":
        list_documents(collection)
    elif args.command == "inspect":
        inspect_document(collection, args.pattern, args.num, args.random)
    elif args.command == "search":
        search_content(collection, args.query, args.num)
    elif args.command == "export":
        export_chunks(collection, args.pattern, args.output)
    elif args.command == "stats":
        chunk_statistics(collection)
    elif args.command == "coverage":
        verify_coverage(collection, args.pattern, args.compare)


if __name__ == "__main__":
    main()
