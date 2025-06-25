"""
Logging configuration for thesis project.

This module provides centralized logging configuration with appropriate
formatters, handlers, and log levels for different components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    include_timestamp: bool = True,
    include_module: bool = True
) -> None:
    """
    Configure logging for the thesis project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        include_timestamp: Whether to include timestamps in log messages
        include_module: Whether to include module names in log messages
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    format_parts = []
    if include_timestamp:
        format_parts.append("%(asctime)s")
    if include_module:
        format_parts.append("%(name)s")
    format_parts.extend(["%(levelname)s", "%(message)s"])
    
    formatter = logging.Formatter(" - ".join(format_parts))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    root_logger.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for experiment logging
def log_experiment_start(logger: logging.Logger, config: dict) -> None:
    """Log the start of an experiment with configuration details."""
    logger.info("🚀 Starting experiment")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_experiment_progress(logger: logging.Logger, step: str, progress: float) -> None:
    """Log experiment progress."""
    logger.info(f"📊 {step}: {progress:.1%} complete")


def log_experiment_result(logger: logging.Logger, metric: str, value: float) -> None:
    """Log a key experiment result."""
    logger.info(f"📈 {metric}: {value:.4f}")


def log_experiment_complete(logger: logging.Logger, output_path: Path) -> None:
    """Log experiment completion."""
    logger.info(f"✅ Experiment complete. Results saved to: {output_path}")


def log_performance_warning(logger: logging.Logger, operation: str, duration: float) -> None:
    """Log performance warnings for slow operations."""
    if duration > 60:  # More than 1 minute
        logger.warning(f"⚠️  Slow operation: {operation} took {duration:.1f}s")


def log_memory_usage(logger: logging.Logger, operation: str, memory_mb: float) -> None:
    """Log memory usage information."""
    if memory_mb > 1000:  # More than 1GB
        logger.warning(f"🧠 High memory usage: {operation} using {memory_mb:.1f}MB")
    else:
        logger.debug(f"🧠 Memory usage: {operation} using {memory_mb:.1f}MB")