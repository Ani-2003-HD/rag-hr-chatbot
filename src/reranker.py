"""
Re-ranking module implementing BM25 and cosine scoring for improved retrieval.
"""

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class Reranker:
    """Implements BM25 and cosine similarity re-ranking for search results."""
    
    def __init__(self):
        """Initialize the reranker."""
        self.bm25 = None
        self.tfidf_vectorizer = None
        self.corpus_tfidf = None
        self.corpus_texts = []
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for BM25 and TF-IDF.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.preprocess_text(text).split()
    
    def build_bm25_index(self, documents: List[Dict[str, str]]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dictionaries
        """
        self.corpus_texts = [doc['text'] for doc in documents]
        tokenized_corpus = [self.tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built with {len(documents)} documents")
    
    def build_tfidf_index(self, documents: List[Dict[str, str]]):
        """
        Build TF-IDF index from documents.
        
        Args:
            documents: List of document dictionaries
        """
        self.corpus_texts = [doc['text'] for doc in documents]
        preprocessed_corpus = [self.preprocess_text(text) for text in self.corpus_texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.corpus_tfidf = self.tfidf_vectorizer.fit_transform(preprocessed_corpus)
        print(f"TF-IDF index built with {len(documents)} documents")
    
    def bm25_score(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get BM25 scores for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_bm25_index first.")
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
    
    def tfidf_cosine_score(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get TF-IDF cosine similarity scores for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if self.tfidf_vectorizer is None or self.corpus_tfidf is None:
            raise ValueError("TF-IDF index not built. Call build_tfidf_index first.")
        
        # Transform query
        preprocessed_query = self.preprocess_text(query)
        query_tfidf = self.tfidf_vectorizer.transform([preprocessed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_tfidf, self.corpus_tfidf).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]
    
    def hybrid_rerank(self, query: str, faiss_results: List[Dict], 
                     bm25_weight: float = 0.3, tfidf_weight: float = 0.2, 
                     faiss_weight: float = 0.5, k: int = 5) -> List[Dict]:
        """
        Hybrid re-ranking combining FAISS, BM25, and TF-IDF scores.
        
        Args:
            query: Search query
            faiss_results: Results from FAISS similarity search
            bm25_weight: Weight for BM25 scores
            tfidf_weight: Weight for TF-IDF scores
            faiss_weight: Weight for FAISS scores
            k: Number of final results to return
            
        Returns:
            Re-ranked results
        """
        if not faiss_results:
            return []
        
        # Get document indices from FAISS results
        doc_indices = []
        for result in faiss_results:
            # Find document index by matching text content
            for i, doc_text in enumerate(self.corpus_texts):
                if result['text'] == doc_text:
                    doc_indices.append(i)
                    break
        
        if not doc_indices:
            return faiss_results[:k]
        
        # Get BM25 scores
        bm25_scores = {}
        try:
            bm25_results = self.bm25_score(query, k=len(self.corpus_texts))
            bm25_scores = {idx: score for idx, score in bm25_results}
        except:
            pass
        
        # Get TF-IDF scores
        tfidf_scores = {}
        try:
            tfidf_results = self.tfidf_cosine_score(query, k=len(self.corpus_texts))
            tfidf_scores = {idx: score for idx, score in tfidf_results}
        except:
            pass
        
        # Calculate hybrid scores
        hybrid_results = []
        for i, result in enumerate(faiss_results):
            if i < len(doc_indices):
                doc_idx = doc_indices[i]
                
                # Normalize FAISS score (higher is better for cosine similarity)
                faiss_score = result['score']
                
                # Get BM25 score (0 if not found)
                bm25_score = bm25_scores.get(doc_idx, 0)
                
                # Get TF-IDF score (0 if not found)
                tfidf_score = tfidf_scores.get(doc_idx, 0)
                
                # Calculate hybrid score
                hybrid_score = (
                    faiss_weight * faiss_score +
                    bm25_weight * bm25_score +
                    tfidf_weight * tfidf_score
                )
                
                result_copy = result.copy()
                result_copy['hybrid_score'] = hybrid_score
                result_copy['bm25_score'] = bm25_score
                result_copy['tfidf_score'] = tfidf_score
                result_copy['original_faiss_score'] = faiss_score
                
                hybrid_results.append(result_copy)
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return hybrid_results[:k]
    
    def simple_rerank(self, query: str, faiss_results: List[Dict], 
                     method: str = "bm25", k: int = 5) -> List[Dict]:
        """
        Simple re-ranking using either BM25 or TF-IDF.
        
        Args:
            query: Search query
            faiss_results: Results from FAISS similarity search
            method: Re-ranking method ("bm25" or "tfidf")
            k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not faiss_results:
            return []
        
        # Get document indices from FAISS results
        doc_indices = []
        for result in faiss_results:
            for i, doc_text in enumerate(self.corpus_texts):
                if result['text'] == doc_text:
                    doc_indices.append(i)
                    break
        
        if not doc_indices:
            return faiss_results[:k]
        
        # Get scores based on method
        if method == "bm25":
            try:
                scores = self.bm25_score(query, k=len(self.corpus_texts))
                score_dict = {idx: score for idx, score in scores}
            except:
                return faiss_results[:k]
        elif method == "tfidf":
            try:
                scores = self.tfidf_cosine_score(query, k=len(self.corpus_texts))
                score_dict = {idx: score for idx, score in scores}
            except:
                return faiss_results[:k]
        else:
            return faiss_results[:k]
        
        # Re-rank results
        reranked_results = []
        for i, result in enumerate(faiss_results):
            if i < len(doc_indices):
                doc_idx = doc_indices[i]
                rerank_score = score_dict.get(doc_idx, 0)
                
                result_copy = result.copy()
                result_copy['rerank_score'] = rerank_score
                result_copy['rerank_method'] = method
                
                reranked_results.append(result_copy)
        
        # Sort by re-rank score
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked_results[:k]


if __name__ == "__main__":
    # Test the reranker
    from document_processor import DocumentProcessor
    from embeddings_manager import EmbeddingsManager
    
    # Process document
    processor = DocumentProcessor()
    pdf_path = "HR-Policy (1).pdf"
    
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        
        # Create embeddings and FAISS index
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.generate_embeddings(chunks)
        vectorstore = embeddings_manager.create_faiss_index(chunks, embeddings)
        
        # Build reranker indices
        reranker = Reranker()
        reranker.build_bm25_index(chunks)
        reranker.build_tfidf_index(chunks)
        
        # Test search and reranking
        query = "What is the leave policy?"
        faiss_results = embeddings_manager.similarity_search(query, k=10)
        
        print(f"\nOriginal FAISS results for: '{query}'")
        for i, result in enumerate(faiss_results[:3]):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(result['text'][:200] + "...")
        
        # Test hybrid reranking
        hybrid_results = reranker.hybrid_rerank(query, faiss_results, k=3)
        
        print(f"\nHybrid re-ranked results:")
        for i, result in enumerate(hybrid_results):
            print(f"\nResult {i+1} (Hybrid Score: {result['hybrid_score']:.4f}):")
            print(f"FAISS: {result['original_faiss_score']:.4f}, "
                  f"BM25: {result['bm25_score']:.4f}, "
                  f"TF-IDF: {result['tfidf_score']:.4f}")
            print(result['text'][:200] + "...")
    else:
        print(f"PDF file not found: {pdf_path}")

