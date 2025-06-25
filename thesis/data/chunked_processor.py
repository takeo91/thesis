"""
Memory-efficient chunked data processing for large datasets.

This module provides utilities for processing large datasets in manageable chunks
to avoid memory exhaustion while maintaining computational efficiency.

Key features:
- Configurable chunk sizes based on available memory
- Streaming data processing for datasets larger than RAM
- Progress monitoring and memory usage tracking
- Graceful fallback to smaller chunks if memory limits exceeded
"""

from __future__ import annotations
import gc
import psutil
from pathlib import Path
from typing import Iterator, List, Optional, Callable, Any, Dict, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

from thesis.core.constants import DEFAULT_CHUNK_SIZE
from thesis.core.logging_config import get_logger, log_memory_usage
from thesis.core.validation import DataValidationError

logger = get_logger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management during chunked processing."""
    max_memory_gb: float = 4.0  # Maximum memory usage in GB
    chunk_size: int = DEFAULT_CHUNK_SIZE  # Default chunk size
    min_chunk_size: int = 100  # Minimum chunk size before failing
    memory_check_interval: int = 10  # Check memory every N chunks
    gc_threshold: float = 0.8  # Trigger garbage collection at this memory fraction


class ChunkedDataProcessor:
    """Memory-efficient processor for large datasets."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the chunked processor.
        
        Args:
            config: Memory configuration settings
        """
        self.config = config or MemoryConfig()
        self.current_chunk_size = self.config.chunk_size
        self.processed_chunks = 0
        
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb / 1024
    
    def check_memory_and_adjust(self) -> None:
        """Check memory usage and adjust chunk size if needed."""
        current_memory = self.get_memory_usage_gb()
        
        if current_memory > self.config.max_memory_gb:
            # Reduce chunk size to manage memory
            new_chunk_size = max(
                self.current_chunk_size // 2,
                self.config.min_chunk_size
            )
            
            if new_chunk_size < self.config.min_chunk_size:
                raise MemoryError(
                    f"Memory usage {current_memory:.2f}GB exceeds limit "
                    f"{self.config.max_memory_gb:.2f}GB and cannot reduce chunk size further"
                )
            
            logger.warning(
                f"High memory usage ({current_memory:.2f}GB), "
                f"reducing chunk size from {self.current_chunk_size} to {new_chunk_size}"
            )
            self.current_chunk_size = new_chunk_size
            
        elif current_memory > self.config.gc_threshold * self.config.max_memory_gb:
            # Trigger garbage collection
            logger.debug("Triggering garbage collection due to high memory usage")
            gc.collect()
    
    def process_array_chunks(
        self,
        data: np.ndarray,
        processor_func: Callable[[np.ndarray], Any],
        axis: int = 0,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Any]:
        """
        Process a large array in chunks along specified axis.
        
        Args:
            data: Input array to process
            processor_func: Function to apply to each chunk
            axis: Axis along which to chunk the data
            progress_callback: Optional progress callback
            
        Returns:
            List of results from processing each chunk
        """
        if data.size == 0:
            return []
        
        data_length = data.shape[axis]
        results = []
        
        for chunk_start in range(0, data_length, self.current_chunk_size):
            chunk_end = min(chunk_start + self.current_chunk_size, data_length)
            
            # Extract chunk
            if axis == 0:
                chunk = data[chunk_start:chunk_end]
            elif axis == 1:
                chunk = data[:, chunk_start:chunk_end]
            else:
                raise ValueError(f"Unsupported axis: {axis}")
            
            # Process chunk
            try:
                result = processor_func(chunk)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_start}:{chunk_end}: {e}")
                raise
            
            self.processed_chunks += 1
            
            # Memory management
            if self.processed_chunks % self.config.memory_check_interval == 0:
                self.check_memory_and_adjust()
                log_memory_usage(logger, "chunk_processing", self.get_memory_usage_gb() * 1024)
            
            # Progress callback
            if progress_callback:
                progress = chunk_end / data_length
                progress_callback(progress)
        
        return results
    
    def process_dataframe_chunks(
        self,
        df: pd.DataFrame,
        processor_func: Callable[[pd.DataFrame], Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Any]:
        """
        Process a large DataFrame in chunks.
        
        Args:
            df: Input DataFrame
            processor_func: Function to apply to each chunk
            progress_callback: Optional progress callback
            
        Returns:
            List of results from processing each chunk
        """
        if len(df) == 0:
            return []
        
        results = []
        total_rows = len(df)
        
        for chunk_start in range(0, total_rows, self.current_chunk_size):
            chunk_end = min(chunk_start + self.current_chunk_size, total_rows)
            
            # Extract chunk
            chunk = df.iloc[chunk_start:chunk_end].copy()
            
            # Process chunk
            try:
                result = processor_func(chunk)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing DataFrame chunk {chunk_start}:{chunk_end}: {e}")
                raise
            
            self.processed_chunks += 1
            
            # Memory management
            if self.processed_chunks % self.config.memory_check_interval == 0:
                self.check_memory_and_adjust()
                log_memory_usage(logger, "dataframe_processing", self.get_memory_usage_gb() * 1024)
            
            # Progress callback
            if progress_callback:
                progress = chunk_end / total_rows
                progress_callback(progress)
        
        return results
    
    def process_file_chunks(
        self,
        file_path: Path,
        processor_func: Callable[[pd.DataFrame], Any],
        read_kwargs: Optional[Dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Any]:
        """
        Process a large file in chunks without loading entire file to memory.
        
        Args:
            file_path: Path to file to process
            processor_func: Function to apply to each chunk
            read_kwargs: Additional kwargs for pd.read_csv
            progress_callback: Optional progress callback
            
        Returns:
            List of results from processing each chunk
        """
        file_path = Path(file_path)
        read_kwargs = read_kwargs or {}
        
        if not file_path.exists():
            raise DataValidationError(f"File not found: {file_path}")
        
        # Get file size for progress tracking
        file_size_bytes = file_path.stat().st_size
        processed_bytes = 0
        results = []
        
        try:
            # Read file in chunks
            chunk_iter = pd.read_csv(
                file_path,
                chunksize=self.current_chunk_size,
                **read_kwargs
            )
            
            for chunk_num, chunk in enumerate(chunk_iter):
                # Process chunk
                try:
                    result = processor_func(chunk)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing file chunk {chunk_num}: {e}")
                    raise
                
                self.processed_chunks += 1
                
                # Estimate progress (rough approximation)
                estimated_bytes = len(chunk) * (file_size_bytes / 1000000)  # Rough estimate
                processed_bytes += estimated_bytes
                
                # Memory management
                if self.processed_chunks % self.config.memory_check_interval == 0:
                    self.check_memory_and_adjust()
                    log_memory_usage(logger, "file_processing", self.get_memory_usage_gb() * 1024)
                
                # Progress callback
                if progress_callback:
                    progress = min(processed_bytes / file_size_bytes, 1.0)
                    progress_callback(progress)
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
        
        return results


class StreamingNDGProcessor:
    """Streaming processor for NDG computations on large datasets."""
    
    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        """Initialize streaming NDG processor."""
        self.chunked_processor = ChunkedDataProcessor(memory_config)
    
    def compute_ndg_streaming(
        self,
        sensor_data: np.ndarray,
        x_values: np.ndarray,
        sigma: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> np.ndarray:
        """
        Compute NDG for large sensor data using streaming approach.
        
        Args:
            sensor_data: Large sensor data array
            x_values: Evaluation points
            sigma: Kernel bandwidth
            progress_callback: Progress callback
            
        Returns:
            NDG values at x_values
        """
        from thesis.fuzzy.membership import compute_ndg_streaming
        
        if sensor_data.size == 0:
            return np.zeros_like(x_values)
        
        # Process data in chunks to avoid memory issues
        def ndg_chunk_processor(data_chunk: np.ndarray) -> np.ndarray:
            """Process a chunk of sensor data."""
            return compute_ndg_streaming(data_chunk, x_values, sigma)
        
        # Process chunks
        chunk_results = self.chunked_processor.process_array_chunks(
            sensor_data,
            ndg_chunk_processor,
            axis=0,
            progress_callback=progress_callback
        )
        
        # Combine results (average across chunks)
        if chunk_results:
            combined_result = np.mean(chunk_results, axis=0)
            return combined_result
        else:
            return np.zeros_like(x_values)


class BatchSimilarityMatrixProcessor:
    """Memory-efficient processor for large similarity matrices."""
    
    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        """Initialize batch similarity matrix processor."""
        self.chunked_processor = ChunkedDataProcessor(memory_config)
    
    def compute_large_similarity_matrix(
        self,
        membership_functions: List[np.ndarray],
        similarity_func: Callable[[np.ndarray, np.ndarray], float],
        progress_callback: Optional[Callable[[float], None]] = None,
        output_file: Optional[Path] = None
    ) -> np.ndarray:
        """
        Compute similarity matrix for large number of membership functions.
        
        Args:
            membership_functions: List of membership function arrays
            similarity_func: Function to compute similarity between two functions
            progress_callback: Progress callback
            output_file: Optional file to save intermediate results
            
        Returns:
            Similarity matrix
        """
        n = len(membership_functions)
        
        if n == 0:
            return np.array([])
        
        # Pre-allocate result matrix
        similarity_matrix = np.zeros((n, n))
        
        total_pairs = n * (n + 1) // 2  # Upper triangle + diagonal
        processed_pairs = 0
        
        # Process in chunks to manage memory
        chunk_size = self.chunked_processor.current_chunk_size
        
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            
            for i in range(i_start, i_end):
                mu_i = membership_functions[i]
                
                # Process j values in chunks
                for j_start in range(i, n, chunk_size):
                    j_end = min(j_start + chunk_size, n)
                    
                    for j in range(max(j_start, i), j_end):
                        mu_j = membership_functions[j]
                        
                        # Compute similarity
                        sim = similarity_func(mu_i, mu_j)
                        similarity_matrix[i, j] = sim
                        
                        # Use symmetry
                        if i != j:
                            similarity_matrix[j, i] = sim
                        
                        processed_pairs += 1
                
                # Progress callback
                if progress_callback:
                    progress = processed_pairs / total_pairs
                    progress_callback(progress)
                
                # Memory check
                if i % self.chunked_processor.config.memory_check_interval == 0:
                    self.chunked_processor.check_memory_and_adjust()
            
            # Save intermediate results if requested
            if output_file and i_end < n:
                logger.debug(f"Saving intermediate results at row {i_end}")
                np.save(f"{output_file}_temp_{i_end}.npy", similarity_matrix)
        
        return similarity_matrix


# Convenience functions
def create_chunked_processor(max_memory_gb: float = 4.0) -> ChunkedDataProcessor:
    """Create a chunked data processor with specified memory limit."""
    config = MemoryConfig(max_memory_gb=max_memory_gb)
    return ChunkedDataProcessor(config)


def process_large_array(
    data: np.ndarray,
    processor_func: Callable[[np.ndarray], Any],
    max_memory_gb: float = 4.0,
    progress_callback: Optional[Callable[[float], None]] = None
) -> List[Any]:
    """
    Convenience function to process large arrays in memory-efficient chunks.
    
    Args:
        data: Large array to process
        processor_func: Function to apply to each chunk  
        max_memory_gb: Maximum memory usage limit
        progress_callback: Optional progress callback
        
    Returns:
        List of results from processing chunks
    """
    processor = create_chunked_processor(max_memory_gb)
    return processor.process_array_chunks(
        data, processor_func, progress_callback=progress_callback
    )