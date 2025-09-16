#!/usr/bin/env python3
"""
Setup script for the RAG HR Chatbot.
This script initializes the system by processing the HR policy document.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from document_processor import DocumentProcessor
from embeddings_manager import EmbeddingsManager
from reranker import Reranker


def setup_rag_system(pdf_path: str, force_rebuild: bool = False):
    """
    Set up the RAG system by processing the document and building indices.
    
    Args:
        pdf_path: Path to the HR policy PDF
        force_rebuild: Whether to rebuild existing indices
    """
    print("🚀 Setting up RAG HR Chatbot system...")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return False
    
    # Check if indices already exist
    if os.path.exists("faiss_index") and not force_rebuild:
        print("✅ FAISS index already exists. Use --force-rebuild to rebuild.")
        return True
    
    try:
        # Step 1: Process document
        print("📄 Processing HR policy document...")
        processor = DocumentProcessor()
        chunks = processor.process_document(pdf_path)
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Step 2: Generate embeddings
        print("🧠 Generating embeddings...")
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.generate_embeddings(chunks)
        print(f"✅ Generated embeddings for {len(embeddings)} chunks")
        
        # Step 3: Build FAISS index
        print("🔍 Building FAISS index...")
        vectorstore = embeddings_manager.create_faiss_index(chunks, embeddings)
        embeddings_manager.save_index("faiss_index")
        print("✅ FAISS index built and saved")
        
        # Step 4: Build reranker indices
        print("🔄 Building reranker indices...")
        reranker = Reranker()
        reranker.build_bm25_index(chunks)
        reranker.build_tfidf_index(chunks)
        print("✅ Reranker indices built")
        
        print("🎉 RAG system setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup RAG HR Chatbot system")
    parser.add_argument(
        "--pdf-path", 
        default="HR-Policy (1).pdf",
        help="Path to the HR policy PDF file"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of existing indices"
    )
    
    args = parser.parse_args()
    
    success = setup_rag_system(args.pdf_path, args.force_rebuild)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

