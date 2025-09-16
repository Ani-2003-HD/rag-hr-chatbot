"""
Streamlit frontend for the RAG HR Chatbot.
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time


# Configuration
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_ENDPOINTS = {
    "query": f"{BACKEND_URL}/query",
    "health": f"{BACKEND_URL}/health",
    "cache_stats": f"{BACKEND_URL}/cache/stats",
    "clear_cache": f"{BACKEND_URL}/cache/clear"
}


def check_backend_health() -> bool:
    """Check if the backend is healthy and initialized."""
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("initialized", False)
        return False
    except requests.exceptions.RequestException:
        return False


def query_hr_policy(question: str, k_retrieve: int = 10, k_rerank: int = 5, 
                   rerank_method: str = "hybrid", use_cache: bool = True) -> Dict[str, Any]:
    """
    Query the HR policy through the backend API.
    
    Args:
        question: The question to ask
        k_retrieve: Number of documents to retrieve
        k_rerank: Number of documents to return after re-ranking
        rerank_method: Re-ranking method
        use_cache: Whether to use cache
        
    Returns:
        Response from the API
    """
    try:
        params = {
            "question": question,
            "k_retrieve": k_retrieve,
            "k_rerank": k_rerank,
            "rerank_method": rerank_method,
            "use_cache": use_cache
        }
        
        response = requests.get(API_ENDPOINTS["query"], params=params, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics from the backend."""
    try:
        response = requests.get(API_ENDPOINTS["cache_stats"], timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": "Failed to get cache stats"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}


def clear_cache() -> bool:
    """Clear the cache."""
    try:
        response = requests.post(API_ENDPOINTS["clear_cache"], timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def display_source(source: Dict[str, Any], index: int):
    """Display a source document."""
    with st.expander(f"Source {index + 1} (Score: {source.get('score', 0):.3f})"):
        st.text(source.get('text', ''))
        if source.get('metadata'):
            st.json(source['metadata'])


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="HR Policy Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🤖 HR Policy Chatbot")
    st.markdown("Ask questions about your HR policy and get intelligent answers powered by RAG (Retrieval-Augmented Generation).")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Backend health check
        st.subheader("Backend Status")
        if check_backend_health():
            st.success("✅ Backend is healthy and initialized")
        else:
            st.error("❌ Backend is not available or not initialized")
            st.info("Make sure the backend is running on http://localhost:8000")
            return
        
        # Query parameters
        st.subheader("Query Parameters")
        k_retrieve = st.slider("Documents to retrieve", 5, 20, 10)
        k_rerank = st.slider("Documents to return", 3, 10, 5)
        rerank_method = st.selectbox(
            "Re-ranking method",
            ["hybrid", "bm25", "tfidf"],
            help="Hybrid combines FAISS, BM25, and TF-IDF scores"
        )
        use_cache = st.checkbox("Use cache", value=True, help="Cache results for faster repeated queries")
        
        # Cache management
        st.subheader("Cache Management")
        if st.button("📊 View Cache Stats"):
            stats = get_cache_stats()
            if "error" not in stats:
                st.json(stats)
            else:
                st.error(stats["error"])
        
        if st.button("🗑️ Clear Cache"):
            if clear_cache():
                st.success("Cache cleared successfully!")
            else:
                st.error("Failed to clear cache")
    
    # Main chat interface
    st.subheader("💬 Ask a Question")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    # Get question from query params or text input
    default_question = st.query_params.get("question", "")
    question = st.text_input(
        "Enter your question about HR policy:",
        value=default_question,
        placeholder="e.g., What is the leave policy? How many vacation days do I get?",
        key="question_input"
    )
    
    # Submit button
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("Ask", type="primary", use_container_width=True)
    
    # Process question
    if submit_button and question:
        with st.spinner("Thinking..."):
            # Query the backend
            result = query_hr_policy(
                question=question,
                k_retrieve=k_retrieve,
                k_rerank=k_rerank,
                rerank_method=rerank_method,
                use_cache=use_cache
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "result": result,
                "timestamp": time.time()
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        
        # Display in reverse order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"**❓ Question:** {chat['question']}")
                
                # Result
                result = chat['result']
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Answer
                    st.markdown(f"**🤖 Answer:** {result.get('answer', 'No answer provided')}")
                    
                    # Metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                    with col2:
                        st.metric("Sources", result.get('num_reranked', 0))
                    with col3:
                        st.metric("Method", result.get('rerank_method', 'N/A'))
                    with col4:
                        cache_status = "✅ Cached" if result.get('from_cache', False) else "🔄 Fresh"
                        st.metric("Cache", cache_status)
                    
                    # Sources
                    if result.get('sources'):
                        st.subheader("📚 Sources")
                        for j, source in enumerate(result['sources']):
                            display_source(source, j)
                
                st.divider()
    
    # Example questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "What is the leave policy?",
        "How many vacation days do employees get?",
        "What is the sick leave policy?",
        "What are the working hours?",
        "What is the dress code policy?",
        "How do I report harassment?",
        "What is the remote work policy?",
        "What are the benefits offered?",
        "What is the performance review process?",
        "What is the code of conduct?"
    ]
    
    # Create columns for example questions
    cols = st.columns(3)
    for i, example in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Use a different approach to set the question
                st.query_params.question = example
                st.rerun()
    
    # Footer
    st.markdown("---")



if __name__ == "__main__":
    main()

