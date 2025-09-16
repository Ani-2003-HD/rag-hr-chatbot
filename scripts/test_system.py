#!/usr/bin/env python3
"""
Test script for the RAG HR Chatbot system.
"""

import os
import sys
import requests
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def test_backend_health():
    """Test backend health endpoint."""
    print("🔍 Testing backend health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is healthy: {data['status']}")
            print(f"   Initialized: {data['initialized']}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False


def test_query_endpoint():
    """Test the query endpoint."""
    print("🔍 Testing query endpoint...")
    
    test_questions = [
        "What is the leave policy?",
        "How many vacation days do employees get?",
        "What is the sick leave policy?",
        "What are the working hours?"
    ]
    
    for question in test_questions:
        print(f"\n📝 Testing question: '{question}'")
        try:
            response = requests.get(
                "http://localhost:8000/query",
                params={"question": question, "k_retrieve": 5, "k_rerank": 3},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Answer: {data['answer'][:100]}...")
                print(f"   Confidence: {data['confidence']:.2f}")
                print(f"   Sources: {len(data['sources'])}")
                print(f"   From cache: {data['from_cache']}")
            else:
                print(f"❌ Query failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Query error: {e}")


def test_cache_endpoints():
    """Test cache-related endpoints."""
    print("🔍 Testing cache endpoints...")
    
    try:
        # Test cache stats
        response = requests.get("http://localhost:8000/cache/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Cache stats: {stats}")
        else:
            print(f"❌ Cache stats failed: {response.status_code}")
        
        # Test cache clear
        response = requests.post("http://localhost:8000/cache/clear", timeout=10)
        if response.status_code == 200:
            print("✅ Cache cleared successfully")
        else:
            print(f"❌ Cache clear failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cache test error: {e}")


def test_frontend():
    """Test if frontend is accessible."""
    print("🔍 Testing frontend...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend not accessible: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend test error: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Starting RAG HR Chatbot system tests...\n")
    
    # Test backend
    if not test_backend_health():
        print("\n❌ Backend is not running. Please start the backend first.")
        print("   Run: python backend.py")
        return
    
    # Test query functionality
    test_query_endpoint()
    
    # Test cache functionality
    test_cache_endpoints()
    
    # Test frontend
    test_frontend()
    
    print("\n🎉 System tests completed!")


if __name__ == "__main__":
    main()

