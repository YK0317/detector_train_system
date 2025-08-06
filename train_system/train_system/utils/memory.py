#!/usr/bin/env python3
"""
Memory optimization utilities for train_system
"""

import gc
import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def optimize_memory():
    """Optimize memory usage by clearing caches and collecting garbage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics"""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "cached_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "device_name": torch.cuda.get_device_name(),
            "device_count": torch.cuda.device_count(),
        }
    return {"cpu_only": True}


def log_memory_usage(stage: str = ""):
    """Log current memory usage"""
    memory_info = get_memory_usage()
    if "cpu_only" in memory_info:
        logger.info(f"🧠 Memory ({stage}): CPU only")
    else:
        logger.info(
            f"🧠 Memory ({stage}): "
            f"Allocated: {memory_info['allocated_mb']:.1f}MB, "
            f"Cached: {memory_info['cached_mb']:.1f}MB"
        )


def move_data_to_device(data, device: torch.device, non_blocking: bool = True):
    """Efficiently move data to device"""
    if isinstance(data, dict):
        return {
            key: move_data_to_device(value, device, non_blocking)
            for key, value in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return [move_data_to_device(item, device, non_blocking) for item in data]
    elif hasattr(data, "to"):
        return data.to(device, non_blocking=non_blocking)
    else:
        return data


def setup_cuda_optimizations():
    """Setup CUDA optimizations for better performance"""
    if torch.cuda.is_available():
        # Enable optimized attention if available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✅ Enabled TF32 optimizations")
        except:
            pass

        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("✅ Enabled cuDNN optimizations")
    else:
        logger.info("ℹ️  CUDA not available, using CPU optimizations")


class MemoryTracker:
    """Track memory usage throughout training"""

    def __init__(self):
        self.baseline = None
        self.peak_usage = 0

    def set_baseline(self):
        """Set baseline memory usage"""
        if torch.cuda.is_available():
            self.baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

    def check_peak(self):
        """Check and update peak memory usage"""
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated()
            if current_peak > self.peak_usage:
                self.peak_usage = current_peak

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        if not torch.cuda.is_available():
            return {"cpu_only": True}

        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()

        stats = {
            "current_mb": current / 1024**2,
            "peak_mb": peak / 1024**2,
            "baseline_mb": (self.baseline or 0) / 1024**2,
            "increase_mb": (current - (self.baseline or 0)) / 1024**2,
        }

        return stats
