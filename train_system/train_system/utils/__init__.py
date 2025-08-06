"""
Utility functions for Train System

Common utilities and helper functions used throughout the system.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(level=numeric_level, format=log_format, handlers=[])

    # Create logger
    logger = logging.getLogger("train_system")
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get PyTorch device

    Args:
        device_str: Device specification ("auto", "cpu", "cuda", "cuda:0", etc.)

    Returns:
        PyTorch device
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    return device


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: str):
    """
    Save data to YAML file

    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from YAML file

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


def create_directories(directories: List[str]):
    """
    Create multiple directories

    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def clean_directory(directory: str, keep_patterns: Optional[List[str]] = None):
    """
    Clean directory contents, optionally keeping files matching patterns

    Args:
        directory: Directory to clean
        keep_patterns: Optional list of glob patterns to keep
    """
    directory = Path(directory)

    if not directory.exists():
        return

    keep_patterns = keep_patterns or []

    for item in directory.iterdir():
        # Check if item should be kept
        should_keep = False
        for pattern in keep_patterns:
            if item.match(pattern):
                should_keep = True
                break

        if not should_keep:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(bytes_size: int) -> str:
    """
    Format byte size to human readable string

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information

    Returns:
        Dictionary with memory usage stats
    """
    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    stats = {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }

    # Add GPU memory if available
    if torch.cuda.is_available():
        stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

    return stats


def set_random_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_file_path(filepath: str, must_exist: bool = True) -> Path:
    """
    Validate and return Path object for file path

    Args:
        filepath: File path string
        must_exist: Whether file must exist

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
        ValueError: If path is invalid
    """
    path = Path(filepath)

    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return path


def get_model_size(model: torch.nn.Module) -> int:
    """
    Get model size in bytes

    Args:
        model: PyTorch model

    Returns:
        Model size in bytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size


class MetricsTracker:
    """
    Simple metrics tracking utility
    """

    def __init__(self):
        self.metrics = {}

    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_latest(self, key: str, default: Any = None):
        """Get latest value for a metric"""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return default

    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average value for a metric"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0

        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]

        return sum(values) / len(values)

    def get_best(self, key: str, mode: str = "max"):
        """Get best value for a metric"""
        if key not in self.metrics or not self.metrics[key]:
            return None

        values = self.metrics[key]
        return max(values) if mode == "max" else min(values)

    def reset(self):
        """Reset all metrics"""
        self.metrics = {}

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary"""
        return self.metrics.copy()


class Timer:
    """
    Simple timer utility
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer"""
        import time

        self.start_time = time.time()
        self.end_time = None

    def stop(self):
        """Stop the timer"""
        import time

        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.time()

    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        import time

        if self.start_time is None:
            return 0.0

        end_time = self.end_time or time.time()
        return end_time - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
