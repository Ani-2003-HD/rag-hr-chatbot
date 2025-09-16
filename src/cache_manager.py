"""
Caching layer for query results to avoid repeated LLM calls.
"""

import os
import json
import hashlib
import pickle
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import redis
from dotenv import load_dotenv

load_dotenv()


class CacheManager:
    """Manages caching of query results and embeddings."""
    
    def __init__(self, cache_type: str = "file", redis_url: str = None):
        """
        Initialize the cache manager.
        
        Args:
            cache_type: Type of cache ("file" or "redis")
            redis_url: Redis URL for redis cache
        """
        self.cache_type = cache_type
        self.cache_dir = "cache"
        
        if cache_type == "redis":
            try:
                self.redis_client = redis.from_url(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"))
                self.redis_client.ping()  # Test connection
                print("Connected to Redis cache")
            except Exception as e:
                print(f"Failed to connect to Redis: {e}. Falling back to file cache.")
                self.cache_type = "file"
                self.redis_client = None
        
        if cache_type == "file":
            os.makedirs(self.cache_dir, exist_ok=True)
            print("Using file-based cache")
    
    def _generate_key(self, query: str, context: str = "") -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: Search query
            context: Additional context for the key
            
        Returns:
            Cache key
        """
        key_string = f"{query}_{context}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_file_path(self, key: str) -> str:
        """
        Get file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            File path
        """
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, query: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.
        
        Args:
            query: Search query
            context: Additional context
            
        Returns:
            Cached result or None
        """
        key = self._generate_key(query, context)
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                file_path = self._get_file_path(key)
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
        except Exception as e:
            print(f"Error retrieving from cache: {e}")
        
        return None
    
    def set(self, query: str, result: Dict[str, Any], context: str = "", 
            ttl: int = 3600) -> bool:
        """
        Cache a result for a query.
        
        Args:
            query: Search query
            result: Result to cache
            context: Additional context
            ttl: Time to live in seconds (for Redis)
            
        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key(query, context)
        
        # Add timestamp to result
        result['cached_at'] = datetime.now().isoformat()
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                self.redis_client.setex(key, ttl, json.dumps(result))
            else:
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    pickle.dump(result, f)
            return True
        except Exception as e:
            print(f"Error caching result: {e}")
            return False
    
    def delete(self, query: str, context: str = "") -> bool:
        """
        Delete cached result for a query.
        
        Args:
            query: Search query
            context: Additional context
            
        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key(query, context)
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                self.redis_client.delete(key)
            else:
                file_path = self._get_file_path(key)
                if os.path.exists(file_path):
                    os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting from cache: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all cached results.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.cache_type == "redis" and self.redis_client:
                self.redis_client.flushdb()
            else:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        stats = {
            'cache_type': self.cache_type,
            'total_entries': 0,
            'cache_size_mb': 0
        }
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                stats['total_entries'] = self.redis_client.dbsize()
                info = self.redis_client.info('memory')
                stats['cache_size_mb'] = info.get('used_memory', 0) / (1024 * 1024)
            else:
                cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
                stats['total_entries'] = len(cache_files)
                
                total_size = 0
                for filename in cache_files:
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
                stats['cache_size_mb'] = total_size / (1024 * 1024)
        except Exception as e:
            print(f"Error getting cache stats: {e}")
        
        return stats
    
    def is_cached(self, query: str, context: str = "") -> bool:
        """
        Check if a query is cached.
        
        Args:
            query: Search query
            context: Additional context
            
        Returns:
            True if cached, False otherwise
        """
        return self.get(query, context) is not None


class QueryCache:
    """Specialized cache for query results with metadata."""
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize query cache.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager
    
    def get_query_result(self, query: str, search_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.
        
        Args:
            query: Search query
            search_params: Search parameters for context
            
        Returns:
            Cached result or None
        """
        context = json.dumps(search_params or {}, sort_keys=True)
        return self.cache_manager.get(query, context)
    
    def cache_query_result(self, query: str, result: Dict[str, Any], 
                          search_params: Dict[str, Any] = None, ttl: int = 3600) -> bool:
        """
        Cache query result.
        
        Args:
            query: Search query
            result: Result to cache
            search_params: Search parameters for context
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        context = json.dumps(search_params or {}, sort_keys=True)
        return self.cache_manager.set(query, result, context, ttl)
    
    def invalidate_query(self, query: str, search_params: Dict[str, Any] = None) -> bool:
        """
        Invalidate cached query result.
        
        Args:
            query: Search query
            search_params: Search parameters for context
            
        Returns:
            True if successful, False otherwise
        """
        context = json.dumps(search_params or {}, sort_keys=True)
        return self.cache_manager.delete(query, context)


if __name__ == "__main__":
    # Test the cache manager
    cache_manager = CacheManager(cache_type="file")
    query_cache = QueryCache(cache_manager)
    
    # Test caching
    test_query = "What is the leave policy?"
    test_result = {
        'answer': 'The leave policy allows employees to take vacation days...',
        'sources': ['chunk_1', 'chunk_2'],
        'confidence': 0.85
    }
    
    # Cache the result
    success = query_cache.cache_query_result(test_query, test_result)
    print(f"Caching result: {'Success' if success else 'Failed'}")
    
    # Retrieve the result
    cached_result = query_cache.get_query_result(test_query)
    if cached_result:
        print(f"Retrieved cached result: {cached_result['answer'][:50]}...")
    else:
        print("No cached result found")
    
    # Get cache stats
    stats = cache_manager.get_cache_stats()
    print(f"Cache stats: {stats}")

