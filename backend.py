"""
FastAPI backend for the RAG HR Chatbot.
"""

import os
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor
from src.embeddings_manager import EmbeddingsManager
from src.reranker import Reranker
from src.cache_manager import CacheManager, QueryCache
from src.rag_pipeline import RAGPipeline

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG HR Chatbot API",
    description="API for HR policy question answering using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the RAG pipeline
rag_pipeline: Optional[RAGPipeline] = None
is_initialized = False


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str
    k_retrieve: Optional[int] = 10
    k_rerank: Optional[int] = 5
    rerank_method: Optional[str] = "hybrid"
    use_cache: Optional[bool] = True


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: list
    confidence: float
    retrieval_method: str
    rerank_method: str
    num_retrieved: int
    num_reranked: int
    from_cache: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    initialized: bool
    pipeline_stats: Optional[Dict[str, Any]] = None


def initialize_rag_pipeline():
    """Initialize the RAG pipeline with HR policy document."""
    global rag_pipeline, is_initialized
    
    try:
        print("Initializing RAG pipeline...")
        
        # Check if PDF exists
        pdf_path = "HR-Policy (1).pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"HR Policy PDF not found: {pdf_path}")
        
        # Process document
        print("Processing HR policy document...")
        processor = DocumentProcessor()
        chunks = processor.process_document(pdf_path)
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.generate_embeddings(chunks)
        
        # Create FAISS index
        print("Building FAISS index...")
        vectorstore = embeddings_manager.create_faiss_index(chunks, embeddings)
        
        # Build reranker indices
        print("Building reranker indices...")
        reranker = Reranker()
        reranker.build_bm25_index(chunks)
        reranker.build_tfidf_index(chunks)
        
        # Initialize cache
        print("Initializing cache...")
        cache_manager = CacheManager()
        query_cache = QueryCache(cache_manager)
        
        # Create RAG pipeline
        print("Creating RAG pipeline...")
        rag_pipeline = RAGPipeline(embeddings_manager, reranker, query_cache)
        
        is_initialized = True
        print("RAG pipeline initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup."""
    try:
        initialize_rag_pipeline()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        # Don't raise exception to allow server to start for debugging


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG HR Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline_stats = None
    if rag_pipeline:
        pipeline_stats = rag_pipeline.get_pipeline_stats()
    
    return HealthResponse(
        status="healthy" if is_initialized else "initializing",
        initialized=is_initialized,
        pipeline_stats=pipeline_stats
    )


@app.post("/query", response_model=QueryResponse)
async def query_hr_policy(request: QueryRequest):
    """
    Query the HR policy for answers.
    
    Args:
        request: Query request with question and parameters
        
    Returns:
        Query response with answer and sources
    """
    if not is_initialized or not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized. Please try again later."
        )
    
    try:
        # Prepare search parameters
        search_params = {
            'k_retrieve': request.k_retrieve,
            'k_rerank': request.k_rerank,
            'rerank_method': request.rerank_method
        }
        
        # Query the RAG pipeline
        result = rag_pipeline.query(
            question=request.question,
            search_params=search_params,
            use_cache=request.use_cache
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/query", response_model=QueryResponse)
async def query_hr_policy_get(
    question: str = Query(..., description="The question to ask about HR policy"),
    k_retrieve: int = Query(10, description="Number of documents to retrieve"),
    k_rerank: int = Query(5, description="Number of documents to return after re-ranking"),
    rerank_method: str = Query("hybrid", description="Re-ranking method: hybrid, bm25, tfidf"),
    use_cache: bool = Query(True, description="Whether to use cache")
):
    """
    Query the HR policy for answers (GET version).
    
    Args:
        question: The question to ask
        k_retrieve: Number of documents to retrieve
        k_rerank: Number of documents to return after re-ranking
        rerank_method: Re-ranking method
        use_cache: Whether to use cache
        
    Returns:
        Query response with answer and sources
    """
    request = QueryRequest(
        question=question,
        k_retrieve=k_retrieve,
        k_rerank=k_rerank,
        rerank_method=rerank_method,
        use_cache=use_cache
    )
    
    return await query_hr_policy(request)


@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        success = rag_pipeline.query_cache.cache_manager.clear_all()
        return {"message": "Cache cleared successfully" if success else "Failed to clear cache"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = rag_pipeline.query_cache.cache_manager.get_cache_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")


@app.post("/reinitialize")
async def reinitialize_pipeline():
    """Reinitialize the RAG pipeline."""
    global is_initialized
    
    try:
        is_initialized = False
        initialize_rag_pipeline()
        return {"message": "RAG pipeline reinitialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reinitializing pipeline: {str(e)}")


if __name__ == "__main__":
    # Run the server
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    
    uvicorn.run(
        "backend:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

