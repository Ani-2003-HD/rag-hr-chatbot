"""
Embeddings and FAISS index management for the RAG system.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


class EmbeddingsManager:
    """Manages embeddings generation and FAISS index operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.sentence_transformer = SentenceTransformer(model_name)
        self.vectorstore = None
        self.documents = []
        
    def generate_embeddings(self, documents: List[Dict[str, str]]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of embedding vectors
        """
        texts = [doc['text'] for doc in documents]
        embeddings = self.sentence_transformer.encode(texts, show_progress_bar=True)
        return embeddings
    
    def create_faiss_index(self, documents: List[Dict[str, str]], embeddings: List[np.ndarray]) -> FAISS:
        """
        Create a FAISS index from documents and embeddings.
        
        Args:
            documents: List of document dictionaries
            embeddings: List of embedding vectors
            
        Returns:
            FAISS vectorstore
        """
        print("Creating FAISS index...")
        
        # Create FAISS index
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add embeddings to index
        index.add(normalized_embeddings.astype('float32'))
        
        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip([doc['text'] for doc in documents], normalized_embeddings)),
            embedding=self.embeddings_model,
            metadatas=documents
        )
        
        self.documents = documents
        print(f"FAISS index created with {len(documents)} documents")
        
        return self.vectorstore
    
    def save_index(self, index_path: str = "faiss_index"):
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
        """
        if self.vectorstore is None:
            raise ValueError("No vectorstore to save. Create index first.")
        
        os.makedirs(index_path, exist_ok=True)
        self.vectorstore.save_local(index_path)
        
        # Save documents metadata
        with open(os.path.join(index_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str = "faiss_index") -> FAISS:
        """
        Load the FAISS index from disk.
        
        Args:
            index_path: Path to load the index from
            
        Returns:
            Loaded FAISS vectorstore
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        self.vectorstore = FAISS.load_local(
            index_path, 
            self.embeddings_model,
            allow_dangerous_deserialization=True
        )
        
        # Load documents metadata
        with open(os.path.join(index_path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        
        print(f"Index loaded from {index_path}")
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform similarity search on the FAISS index.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.vectorstore is None:
            raise ValueError("No vectorstore loaded. Load or create index first.")
        
        # Get query embedding
        query_embedding = self.sentence_transformer.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        docs_with_scores = self.vectorstore.similarity_search_with_score_by_vector(
            query_embedding[0].astype('float32'), 
            k=k
        )
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Dict:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary
        """
        for doc in self.documents:
            if doc['id'] == doc_id:
                return doc
        return None


if __name__ == "__main__":
    # Test the embeddings manager
    from document_processor import DocumentProcessor
    
    # Process document
    processor = DocumentProcessor()
    pdf_path = "HR-Policy (1).pdf"
    
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        
        # Create embeddings
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.generate_embeddings(chunks)
        
        # Create FAISS index
        vectorstore = embeddings_manager.create_faiss_index(chunks, embeddings)
        
        # Test search
        query = "What is the leave policy?"
        results = embeddings_manager.similarity_search(query, k=3)
        
        print(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(result['text'][:200] + "...")
    else:
        print(f"PDF file not found: {pdf_path}")

