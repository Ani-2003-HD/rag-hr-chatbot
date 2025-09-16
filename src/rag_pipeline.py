"""
RAG pipeline that combines retrieval, re-ranking, and generation.
"""

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

from .embeddings_manager import EmbeddingsManager
from .reranker import Reranker
from .cache_manager import QueryCache

load_dotenv()


class RAGPipeline:
    """Main RAG pipeline that orchestrates retrieval, re-ranking, and generation."""
    
    def __init__(self, embeddings_manager: EmbeddingsManager, 
                 reranker: Reranker, query_cache: QueryCache):
        """
        Initialize the RAG pipeline.
        
        Args:
            embeddings_manager: Embeddings manager instance
            reranker: Re-ranker instance
            query_cache: Query cache instance
        """
        self.embeddings_manager = embeddings_manager
        self.reranker = reranker
        self.query_cache = query_cache
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt template
        self.prompt_template = """
You are an HR assistant chatbot. Answer the user's question based on the provided HR policy documents.

Context from HR Policy:
{context}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the answer is not in the context, say "I don't have enough information in the HR policy to answer this question"
3. Be concise but comprehensive
4. If relevant, cite specific sections or policies
5. Use a professional and helpful tone

Answer:
"""
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        return self.embeddings_manager.similarity_search(query, k=k)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               method: str = "hybrid", k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank retrieved documents.
        
        Args:
            query: Search query
            documents: Retrieved documents
            method: Re-ranking method ("hybrid", "bm25", "tfidf")
            k: Number of documents to return
            
        Returns:
            Re-ranked documents
        """
        if method == "hybrid":
            return self.reranker.hybrid_rerank(query, documents, k=k)
        elif method == "bm25":
            return self.reranker.simple_rerank(query, documents, method="bm25", k=k)
        elif method == "tfidf":
            return self.reranker.simple_rerank(query, documents, method="tfidf", k=k)
        else:
            return documents[:k]
    
    def generate_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Generate context string from retrieved documents.
        
        Args:
            documents: Retrieved and re-ranked documents
            
        Returns:
            Context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using Gemini LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        try:
            prompt = self.prompt_template.format(
                context=context,
                question=query
            )
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, question: str, search_params: Dict[str, Any] = None, 
              use_cache: bool = True) -> Dict[str, Any]:
        """
        Main query method that orchestrates the entire RAG pipeline.
        
        Args:
            question: User question
            search_params: Search parameters (k, rerank_method, etc.)
            use_cache: Whether to use cache
            
        Returns:
            Complete response with answer and metadata
        """
        # Default search parameters
        params = {
            'k_retrieve': 10,
            'k_rerank': 5,
            'rerank_method': 'hybrid'
        }
        if search_params:
            params.update(search_params)
        
        # Check cache first
        if use_cache:
            cached_result = self.query_cache.get_query_result(question, params)
            if cached_result:
                cached_result['from_cache'] = True
                return cached_result
        
        # Retrieve documents
        retrieved_docs = self.retrieve(question, k=params['k_retrieve'])
        
        if not retrieved_docs:
            result = {
                'answer': "I don't have enough information in the HR policy to answer this question.",
                'sources': [],
                'confidence': 0.0,
                'retrieval_method': 'faiss',
                'rerank_method': params['rerank_method'],
                'from_cache': False
            }
            
            # Cache the result
            if use_cache:
                self.query_cache.cache_query_result(question, result, params)
            
            return result
        
        # Re-rank documents
        reranked_docs = self.rerank(
            question, 
            retrieved_docs, 
            method=params['rerank_method'],
            k=params['k_rerank']
        )
        
        # Generate context
        context = self.generate_context(reranked_docs)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Prepare sources
        sources = []
        for doc in reranked_docs:
            source_info = {
                'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'metadata': doc.get('metadata', {}),
                'score': doc.get('hybrid_score', doc.get('score', 0))
            }
            sources.append(source_info)
        
        # Calculate confidence based on scores
        if reranked_docs:
            avg_score = sum(doc.get('hybrid_score', doc.get('score', 0)) for doc in reranked_docs) / len(reranked_docs)
            confidence = min(avg_score * 2, 1.0)  # Scale and cap at 1.0
        else:
            confidence = 0.0
        
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'retrieval_method': 'faiss',
            'rerank_method': params['rerank_method'],
            'num_retrieved': len(retrieved_docs),
            'num_reranked': len(reranked_docs),
            'from_cache': False
        }
        
        # Cache the result
        if use_cache:
            self.query_cache.cache_query_result(question, result, params)
        
        return result
    
    def batch_query(self, questions: List[str], search_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            search_params: Search parameters
            
        Returns:
            List of results
        """
        results = []
        for question in questions:
            result = self.query(question, search_params)
            results.append(result)
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Pipeline statistics
        """
        cache_stats = self.query_cache.cache_manager.get_cache_stats()
        
        return {
            'cache_stats': cache_stats,
            'embeddings_model': self.embeddings_manager.model_name,
            'llm_model': 'gemini-pro',
            'reranker_available': self.reranker.bm25 is not None and self.reranker.tfidf_vectorizer is not None
        }


if __name__ == "__main__":
    # Test the RAG pipeline
    from document_processor import DocumentProcessor
    from embeddings_manager import EmbeddingsManager
    from reranker import Reranker
    from cache_manager import CacheManager
    
    # Process document
    processor = DocumentProcessor()
    pdf_path = "HR-Policy (1).pdf"
    
    if os.path.exists(pdf_path):
        chunks = processor.process_document(pdf_path)
        
        # Create components
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.generate_embeddings(chunks)
        vectorstore = embeddings_manager.create_faiss_index(chunks, embeddings)
        
        reranker = Reranker()
        reranker.build_bm25_index(chunks)
        reranker.build_tfidf_index(chunks)
        
        cache_manager = CacheManager()
        query_cache = QueryCache(cache_manager)
        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(embeddings_manager, reranker, query_cache)
        
        # Test query
        question = "What is the leave policy for employees?"
        result = rag_pipeline.query(question)
        
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Sources: {len(result['sources'])}")
        
        # Test cache
        result2 = rag_pipeline.query(question)
        print(f"From cache: {result2['from_cache']}")
        
        # Get stats
        stats = rag_pipeline.get_pipeline_stats()
        print(f"Pipeline stats: {stats}")
    else:
        print(f"PDF file not found: {pdf_path}")

