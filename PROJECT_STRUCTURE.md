# Project Structure

```
data_mites_assignment/
├── 📁 src/                          # Core RAG system modules
│   ├── __init__.py                  # Package initialization
│   ├── document_processor.py        # PDF text extraction and cleaning
│   ├── embeddings_manager.py        # Embeddings generation and FAISS index
│   ├── reranker.py                  # BM25 and TF-IDF re-ranking
│   ├── cache_manager.py             # Query caching layer
│   └── rag_pipeline.py              # Main RAG pipeline orchestration
├── 📁 scripts/                      # Utility scripts
│   ├── setup.py                     # System setup and initialization
│   └── test_system.py               # System testing script
├── 📄 backend.py                    # FastAPI backend server
├── 📄 frontend.py                   # Streamlit frontend application
├── 📄 requirements.txt              # Python dependencies
├── 📄 env_template.txt              # Environment variables template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 Dockerfile                    # Docker container configuration
├── 📄 docker-compose.yml            # Multi-service Docker setup
├── 📄 start.sh                      # Local development startup script
├── 📄 build_docker.sh               # Docker build script
├── 📄 README.md                     # Comprehensive documentation
├── 📄 PROJECT_STRUCTURE.md          # This file
└── 📄 HR-Policy (1).pdf            # HR policy document (input)
```

## Key Components

### Core RAG System (`src/`)
- **Document Processor**: Handles PDF text extraction, cleaning, and chunking
- **Embeddings Manager**: Generates embeddings and manages FAISS vector index
- **Re-ranker**: Implements BM25 and TF-IDF re-ranking for improved retrieval
- **Cache Manager**: Provides caching layer with Redis and file-based options
- **RAG Pipeline**: Orchestrates the complete retrieval-augmentation-generation process

### Backend (`backend.py`)
- FastAPI server with RESTful endpoints
- Health checks and system monitoring
- Query processing with configurable parameters
- Cache management endpoints
- CORS support for frontend integration

### Frontend (`frontend.py`)
- Streamlit-based chat interface
- Real-time query processing
- Source attribution and confidence scoring
- Cache statistics and management
- Example questions and parameter tuning

### Deployment
- **Docker**: Containerized deployment with multi-service setup
- **Docker Compose**: Redis, backend, and frontend services
- **Scripts**: Automated setup and testing utilities

### Generated Files (Runtime)
```
├── 📁 cache/                        # Query cache storage
├── 📁 faiss_index/                  # FAISS vector index
│   ├── index.faiss                  # FAISS index file
│   ├── index.pkl                    # FAISS metadata
│   └── documents.pkl                # Document metadata
└── 📁 embeddings/                   # Embeddings storage (if used)
```

## Data Flow

1. **Document Processing**: PDF → Text Extraction → Cleaning → Chunking
2. **Embedding Generation**: Text Chunks → Sentence Transformers → Vector Embeddings
3. **Index Building**: Embeddings → FAISS Index + BM25/TF-IDF Indices
4. **Query Processing**: User Question → Retrieval → Re-ranking → Context Generation
5. **Answer Generation**: Context + Question → Gemini LLM → Final Answer
6. **Caching**: Results cached for improved performance

## Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **LLM**: Google Gemini Pro
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **Re-ranking**: BM25, TF-IDF
- **Caching**: Redis, File-based
- **Containerization**: Docker, Docker Compose
- **Document Processing**: PyPDF2, LangChain

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /query` - Query HR policy (GET)
- `POST /query` - Query HR policy (POST)
- `GET /cache/stats` - Cache statistics
- `POST /cache/clear` - Clear cache
- `POST /reinitialize` - Reinitialize pipeline

## Configuration

- Environment variables in `.env` file
- Configurable chunk sizes and overlap
- Adjustable retrieval and re-ranking parameters
- Multiple re-ranking methods (hybrid, BM25, TF-IDF)
- Cache TTL and backend selection

