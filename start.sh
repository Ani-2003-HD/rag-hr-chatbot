#!/bin/bash

echo "🚀 Starting RAG HR Chatbot..."

# Start backend in background
echo "🖥️  Starting backend server..."
python backend.py &
BACKEND_PID=$!

# Wait for backend to initialize
echo "⏳ Waiting for backend to initialize..."
sleep 30

# Check if backend is running
for i in {1..10}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is running on http://localhost:8000"
        break
    else
        echo "⏳ Waiting for backend... (attempt $i/10)"
        sleep 5
    fi
done

# Final check
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend failed to start after 50 seconds"
    echo "🔍 Checking backend logs..."
    ps aux | grep python
    exit 1
fi

# Start frontend
echo "🌐 Starting frontend..."
echo "📱 Frontend will be available at http://localhost:8501"
echo "🔗 Backend API will be available at http://localhost:8000"
echo "📚 API documentation at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start frontend (this will block)
streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0
