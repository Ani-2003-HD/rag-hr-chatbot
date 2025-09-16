# RAG HR Chatbot

A sophisticated HR Policy Question-Answering system built with Retrieval-Augmented Generation (RAG) technology. This chatbot can answer questions about HR policies by intelligently retrieving relevant information from policy documents and generating accurate responses using Google's Gemini LLM.

## 📍 Repository Information

- **GitHub Repository**: [https://github.com/Ani-2003-HD/rag-hr-chatbot](https://github.com/Ani-2003-HD/rag-hr-chatbot)
- **Docker Hub**: [aniurddhahd/rag-hr-chatbot](https://hub.docker.com/r/aniurddhahd/rag-hr-chatbot)

## 🚀 Features

- **Document Processing**: Extracts and processes text from HR policy PDFs
- **Advanced Retrieval**: Uses FAISS vector search for efficient similarity matching
- **Intelligent Re-ranking**: Implements BM25 and TF-IDF re-ranking for improved results
- **Caching Layer**: Redis/file-based caching to avoid repeated LLM calls
- **RAG Pipeline**: Combines retrieval, re-ranking, and generation for accurate answers
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Interactive UI**: Streamlit frontend for easy interaction
- **Containerized**: Docker support for easy deployment
- **Source Attribution**: Provides sources for all answers with confidence scores

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   RAG Pipeline  │
│   Frontend      │◄──►│   Backend       │◄──►│                 │
│   (Port 8501)   │    │   (Port 8000)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │   FAISS Index   │
                       │     Cache       │    │   + Embeddings  │
                       │   (Port 6379)   │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Google Gemini API key
- Redis (optional, for caching)

## 🛠️ Installation

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data_mites_assignment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   # Edit .env and add your GEMINI_API_KEY
   ```

5. **Run the application**
   ```bash
   # Start backend
   python backend.py
   
   # In another terminal, start frontend
   streamlit run frontend.py
   ```

### Option 2: Docker Deployment

1. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   # Edit .env and add your GEMINI_API_KEY
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `APP_HOST` | Backend host | `0.0.0.0` |
| `APP_PORT` | Backend port | `8000` |
| `STREAMLIT_PORT` | Frontend port | `8501` |

### Model Configuration

The system uses the following models by default:
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **LLM**: `gemini-pro` (Google Gemini)
- **Vector Store**: FAISS with cosine similarity

## 📚 API Documentation

### Endpoints

#### `GET /health`
Check the health status of the API and RAG pipeline.

**Response:**
```json
{
  "status": "healthy",
  "initialized": true,
  "pipeline_stats": {
    "cache_stats": {...},
    "embeddings_model": "all-MiniLM-L6-v2",
    "llm_model": "gemini-pro"
  }
}
```

#### `GET /query`
Query the HR policy for answers.

**Parameters:**
- `question` (string): The question to ask
- `k_retrieve` (int): Number of documents to retrieve (default: 10)
- `k_rerank` (int): Number of documents to return after re-ranking (default: 5)
- `rerank_method` (string): Re-ranking method - "hybrid", "bm25", "tfidf" (default: "hybrid")
- `use_cache` (boolean): Whether to use cache (default: true)

**Response:**
```json
{
  "answer": "Based on the HR policy...",
  "sources": [
    {
      "text": "Relevant text excerpt...",
      "metadata": {...},
      "score": 0.85
    }
  ],
  "confidence": 0.87,
  "retrieval_method": "faiss",
  "rerank_method": "hybrid",
  "num_retrieved": 10,
  "num_reranked": 5,
  "from_cache": false
}
```

#### `POST /query`
Same as GET /query but with JSON body.

#### `GET /cache/stats`
Get cache statistics.

#### `POST /cache/clear`
Clear the query cache.

#### `POST /reinitialize`
Reinitialize the RAG pipeline.

## 🎯 Usage Examples

### Using the Streamlit Frontend

1. Open http://localhost:8501 in your browser
2. Enter your question in the text input
3. Adjust parameters in the sidebar if needed
4. Click "Ask" to get your answer
5. View sources and confidence scores

### Using the API Directly

```python
import requests

# Query the HR policy
response = requests.get("http://localhost:8000/query", params={
    "question": "What is the leave policy?",
    "k_retrieve": 10,
    "k_rerank": 5,
    "rerank_method": "hybrid"
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

### Example Questions

- "What is the leave policy?"
- "How many vacation days do employees get?"
- "What is the sick leave policy?"
- "What are the working hours?"
- "What is the dress code policy?"
- "How do I report harassment?"
- "What is the remote work policy?"
- "What are the benefits offered?"

## 🔍 Technical Details

### Document Processing Pipeline

1. **PDF Extraction**: Uses PyPDF2 to extract text from HR policy PDFs
2. **Text Cleaning**: Removes special characters, normalizes whitespace
3. **Chunking**: Splits text into manageable chunks (1000 chars, 200 overlap)
4. **Embedding Generation**: Creates vector embeddings using Sentence Transformers

### Retrieval and Re-ranking

1. **FAISS Search**: Initial retrieval using cosine similarity
2. **BM25 Re-ranking**: Keyword-based scoring for better relevance
3. **TF-IDF Re-ranking**: Term frequency-based scoring
4. **Hybrid Scoring**: Combines all methods with configurable weights

### Caching Strategy

- **Query-level Caching**: Caches complete query results
- **Multiple Backends**: Supports Redis and file-based caching
- **TTL Support**: Configurable time-to-live for cached results
- **Cache Invalidation**: Manual cache clearing capabilities

## 🚀 Deployment

### Docker Hub Deployment

1. **Build the image**
   ```bash
   docker build -t aniurddhahd/rag-hr-chatbot .
   ```

2. **Push to Docker Hub**
   ```bash
   docker push aniurddhahd/rag-hr-chatbot
   ```

3. **Run from Docker Hub**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 \
     -e GEMINI_API_KEY=your_api_key \
     aniurddhahd/rag-hr-chatbot
   ```

### Available Docker Images

The following Docker images are available on Docker Hub:

- **Backend**: `aniurddhahd/rag-hr-chatbot-backend:latest`
- **Frontend**: `aniurddhahd/rag-hr-chatbot-frontend:latest`
- **Complete Application**: `aniurddhahd/rag-hr-chatbot:latest`

### Production Considerations

- Use environment variables for sensitive data
- Set up proper logging and monitoring
- Configure reverse proxy (nginx) for production
- Use managed Redis service for caching
- Implement rate limiting and authentication
- Set up health checks and auto-restart

## 🔒 Security Notes

### API Security
- The API is currently open for development
- Implement authentication for production use
- Add rate limiting to prevent abuse
- Use HTTPS in production environments

### Data Privacy
- HR policy documents may contain sensitive information
- Ensure proper access controls
- Consider data encryption at rest
- Implement audit logging for compliance

### Environment Variables
- Never commit API keys to version control
- Use secure secret management in production
- Rotate API keys regularly
- Monitor API usage and costs

## 🐛 Troubleshooting

### Common Issues

1. **Backend not initializing**
   - Check if GEMINI_API_KEY is set correctly
   - Ensure HR-Policy (1).pdf exists in the project root
   - Check logs for specific error messages

2. **Frontend can't connect to backend**
   - Verify backend is running on port 8000
   - Check CORS settings
   - Ensure no firewall blocking the connection

3. **Poor answer quality**
   - Try different re-ranking methods
   - Adjust k_retrieve and k_rerank parameters
   - Check if the question is covered in the HR policy

4. **Slow response times**
   - Enable caching
   - Reduce k_retrieve and k_rerank values
   - Use Redis for better cache performance

### Logs and Debugging

- Backend logs: Check console output when running `python backend.py`
- Frontend logs: Check Streamlit logs in the terminal
- Docker logs: Use `docker-compose logs` to view container logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for the LLM capabilities
- Hugging Face for the sentence transformer models
- FAISS for efficient vector search
- Streamlit and FastAPI for the web framework
- The open-source community for various libraries used

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This is a demonstration project. For production use, ensure proper security measures, testing, and compliance with your organization's policies.

