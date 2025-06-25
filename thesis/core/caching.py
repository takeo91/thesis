"""
Intelligent caching system for expensive computations.

This module provides multi-level caching for NDG computations, similarity metrics,
and other expensive operations with automatic cache management and persistence.

Key features:
- LRU cache for in-memory fast access
- Disk-based persistent cache for large results
- Automatic cache invalidation and cleanup
- Memory usage monitoring and limits
- Cache statistics and performance tracking
"""

from __future__ import annotations
import hashlib
import pickle
import time
from functools import wraps, lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import psutil
from dataclasses import dataclass, field

from thesis.core.constants import DEFAULT_CHUNK_SIZE
from thesis.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    memory_cache_size: int = 128  # LRU cache size
    disk_cache_dir: Path = field(default_factory=lambda: Path("cache"))
    max_disk_cache_gb: float = 1.0  # Maximum disk cache size in GB
    enable_disk_cache: bool = True
    cache_expiry_hours: int = 24  # Cache expiry time
    enable_compression: bool = True  # Compress cached data
    auto_cleanup: bool = True  # Automatic cache cleanup


class CacheStats:
    """Cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.disk_reads = 0
        self.disk_writes = 0
        self.total_computation_time_saved = 0.0
        self.cache_creation_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate cache efficiency score (time saved / total time)."""
        total_time = time.time() - self.cache_creation_time
        return self.total_computation_time_saved / total_time if total_time > 0 else 0.0
    
    def log_stats(self) -> None:
        """Log cache statistics."""
        logger.info(f"Cache Stats - Hit Rate: {self.hit_rate:.2%}, "
                   f"Hits: {self.hits}, Misses: {self.misses}, "
                   f"Time Saved: {self.total_computation_time_saved:.2f}s")


class IntelligentCache:
    """Multi-level intelligent caching system."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the caching system.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # Create cache directory
        if self.config.enable_disk_cache:
            self.config.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory cache
        self._memory_cache = {}
        self._cache_access_times = {}
        
        # Cleanup old cache files if enabled
        if self.config.auto_cleanup:
            self._cleanup_expired_cache()
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from function arguments."""
        # Create a stable hash from arguments
        key_data = []
        
        for arg in args:
            if isinstance(arg, np.ndarray):
                # For numpy arrays, use shape and a sample of data
                key_data.append(f"array_{arg.shape}_{hash(arg.tobytes())}")
            else:
                key_data.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                key_data.append(f"{k}_array_{v.shape}_{hash(v.tobytes())}")
            else:
                key_data.append(f"{k}_{v}")
        
        # Create SHA256 hash of the key data
        key_string = "|".join(key_data)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get the disk cache file path for a given key."""
        return self.config.disk_cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_expired(self, file_path: Path) -> bool:
        """Check if a cache file has expired."""
        if not file_path.exists():
            return True
        
        file_age_hours = (time.time() - file_path.stat().st_mtime) / 3600
        return file_age_hours > self.config.cache_expiry_hours
    
    def _cleanup_expired_cache(self) -> None:
        """Clean up expired cache files."""
        if not self.config.enable_disk_cache:
            return
        
        try:
            for cache_file in self.config.disk_cache_dir.glob("*.pkl"):
                if self._is_cache_expired(cache_file):
                    cache_file.unlink()
                    logger.debug(f"Removed expired cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Error during cache cleanup: {e}")
    
    def _manage_memory_cache_size(self) -> None:
        """Manage memory cache size using LRU eviction."""
        while len(self._memory_cache) > self.config.memory_cache_size:
            # Find least recently used item
            oldest_key = min(self._cache_access_times.keys(), 
                           key=self._cache_access_times.get)
            
            # Remove from cache
            del self._memory_cache[oldest_key]
            del self._cache_access_times[oldest_key]
    
    def _save_to_disk(self, cache_key: str, data: Any) -> None:
        """Save data to disk cache."""
        if not self.config.enable_disk_cache:
            return
        
        try:
            cache_file = self._get_disk_cache_path(cache_key)
            
            # Save data with optional compression
            with open(cache_file, 'wb') as f:
                if self.config.enable_compression:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(data, f)
            
            self.stats.disk_writes += 1
            logger.debug(f"Saved data to disk cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """Load data from disk cache."""
        if not self.config.enable_disk_cache:
            return None
        
        try:
            cache_file = self._get_disk_cache_path(cache_key)
            
            if not cache_file.exists() or self._is_cache_expired(cache_file):
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.stats.disk_reads += 1
            logger.debug(f"Loaded data from disk cache: {cache_file}")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get data from cache (memory first, then disk)."""
        # Check memory cache first
        if cache_key in self._memory_cache:
            self._cache_access_times[cache_key] = time.time()
            self.stats.hits += 1
            return self._memory_cache[cache_key]
        
        # Check disk cache
        data = self._load_from_disk(cache_key)
        if data is not None:
            # Load into memory cache
            self._memory_cache[cache_key] = data
            self._cache_access_times[cache_key] = time.time()
            self._manage_memory_cache_size()
            self.stats.hits += 1
            return data
        
        self.stats.misses += 1
        return None
    
    def put(self, cache_key: str, data: Any) -> None:
        """Store data in cache (memory and optionally disk)."""
        # Store in memory cache
        self._memory_cache[cache_key] = data
        self._cache_access_times[cache_key] = time.time()
        self._manage_memory_cache_size()
        
        # Store in disk cache
        self._save_to_disk(cache_key, data)
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with caching."""
        cache_key = self._generate_cache_key(func.__name__, *args, **kwargs)
        
        # Try to get from cache
        result = self.get(cache_key)
        if result is not None:
            return result
        
        # Compute and cache result
        start_time = time.time()
        result = func(*args, **kwargs)
        computation_time = time.time() - start_time
        
        # Store in cache
        self.put(cache_key, result)
        
        # Update stats
        self.stats.total_computation_time_saved += computation_time
        
        logger.debug(f"Computed and cached {func.__name__} in {computation_time:.4f}s")
        return result
    
    def clear(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._cache_access_times.clear()
        
        if self.config.enable_disk_cache:
            try:
                for cache_file in self.config.disk_cache_dir.glob("*.pkl"):
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Error clearing disk cache: {e}")
    
    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get information about cache sizes."""
        memory_size = len(self._memory_cache)
        
        disk_size_mb = 0
        disk_files = 0
        if self.config.enable_disk_cache and self.config.disk_cache_dir.exists():
            for cache_file in self.config.disk_cache_dir.glob("*.pkl"):
                disk_size_mb += cache_file.stat().st_size / 1024 / 1024
                disk_files += 1
        
        return {
            "memory_cache_entries": memory_size,
            "disk_cache_files": disk_files,
            "disk_cache_size_mb": disk_size_mb,
            "hit_rate": self.stats.hit_rate,
            "total_hits": self.stats.hits,
            "total_misses": self.stats.misses
        }


# Global cache instance
_global_cache = None

def get_cache() -> IntelligentCache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def cached(cache_instance: Optional[IntelligentCache] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or get_cache()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cache.cached_call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def cache_ndg_computation(
    sensor_data: np.ndarray,
    x_values: np.ndarray, 
    sigma: float,
    cache_instance: Optional[IntelligentCache] = None
) -> np.ndarray:
    """
    Cached NDG computation wrapper.
    
    Args:
        sensor_data: Sensor data array
        x_values: Evaluation points
        sigma: Kernel bandwidth
        cache_instance: Optional cache instance
        
    Returns:
        NDG values at x_values
    """
    cache = cache_instance or get_cache()
    
    def ndg_computation():
        from thesis.fuzzy.membership import compute_ndg_streaming
        return compute_ndg_streaming(sensor_data, x_values, sigma)
    
    return cache.cached_call(ndg_computation)


def cache_similarity_matrix(
    membership_functions: list,
    metric_name: str,
    cache_instance: Optional[IntelligentCache] = None
) -> np.ndarray:
    """
    Cached similarity matrix computation.
    
    Args:
        membership_functions: List of membership functions
        metric_name: Name of similarity metric
        cache_instance: Optional cache instance
        
    Returns:
        Similarity matrix
    """
    cache = cache_instance or get_cache()
    
    def matrix_computation():
        from thesis.fuzzy.similarity_optimized import create_batch_processor
        processor = create_batch_processor()
        return processor.compute_pairwise_matrix(
            membership_functions, [metric_name]
        )[metric_name]
    
    return cache.cached_call(matrix_computation)


# Configuration helpers
def configure_cache(
    memory_size: int = 128,
    disk_cache_gb: float = 1.0,
    cache_dir: Optional[Path] = None
) -> IntelligentCache:
    """
    Configure the global cache with custom settings.
    
    Args:
        memory_size: Memory cache size
        disk_cache_gb: Disk cache size limit in GB
        cache_dir: Cache directory path
        
    Returns:
        Configured cache instance
    """
    config = CacheConfig(
        memory_cache_size=memory_size,
        max_disk_cache_gb=disk_cache_gb,
        disk_cache_dir=cache_dir or Path("cache")
    )
    
    global _global_cache
    _global_cache = IntelligentCache(config)
    return _global_cache


def clear_all_caches() -> None:
    """Clear all caches and reset statistics."""
    cache = get_cache()
    cache.clear()
    cache.stats = CacheStats()
    logger.info("All caches cleared")


def print_cache_stats() -> None:
    """Print comprehensive cache statistics."""
    cache = get_cache()
    cache.stats.log_stats()
    
    size_info = cache.get_cache_size_info()
    logger.info(f"Memory cache: {size_info['memory_cache_entries']} entries")
    logger.info(f"Disk cache: {size_info['disk_cache_files']} files, "
               f"{size_info['disk_cache_size_mb']:.1f} MB")


# Specialized caches for different use cases
class NDGCache(IntelligentCache):
    """Specialized cache for NDG computations."""
    
    def __init__(self):
        config = CacheConfig(
            memory_cache_size=64,  # Smaller memory cache for large NDG results
            max_disk_cache_gb=2.0,  # Larger disk cache for NDG data
            cache_expiry_hours=48   # Longer expiry for expensive NDG computations
        )
        super().__init__(config)


class SimilarityCache(IntelligentCache):
    """Specialized cache for similarity computations."""
    
    def __init__(self):
        config = CacheConfig(
            memory_cache_size=256,  # Larger memory cache for fast similarity access
            max_disk_cache_gb=0.5,  # Smaller disk cache for similarity results
            cache_expiry_hours=12   # Shorter expiry for similarity results
        )
        super().__init__(config)