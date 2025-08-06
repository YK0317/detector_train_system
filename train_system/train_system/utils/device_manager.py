"""
Intelligent device detection and management
Fixes hard-coded device settings in the train system
"""

import logging
import platform
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Intelligent device detection and management"""

    @staticmethod
    def get_optimal_device(preferred_device: str = "auto") -> torch.device:
        """
        Get the optimal device based on availability and preference

        Args:
            preferred_device: Preferred device ('auto', 'cpu', 'cuda', 'mps', 'cuda:N')

        Returns:
            torch.device object for the optimal device
        """

        if preferred_device == "auto":
            return DeviceManager._auto_detect_device()

        elif preferred_device.startswith("cuda"):
            return DeviceManager._get_cuda_device(preferred_device)

        elif preferred_device == "mps":
            return DeviceManager._get_mps_device()

        else:  # cpu or any other
            logger.info("Using CPU device")
            return torch.device("cpu")

    @staticmethod
    def _auto_detect_device() -> torch.device:
        """Auto-detect the best available device"""

        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                # Select device with most free memory
                best_device = DeviceManager._get_best_cuda_device()
                logger.info(
                    f"Selected CUDA device {best_device} out of {device_count} available"
                )
                return torch.device(f"cuda:{best_device}")
            else:
                logger.info("Using CUDA device 0")
                return torch.device("cuda:0")

        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) device")
            return torch.device("mps")

        # Fallback to CPU
        else:
            logger.info("CUDA/MPS not available, using CPU")
            return torch.device("cpu")

    @staticmethod
    def _get_cuda_device(preferred_device: str) -> torch.device:
        """Get CUDA device with fallback"""

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")

        # Extract device number
        if ":" in preferred_device:
            device_num = int(preferred_device.split(":")[-1])
        else:
            device_num = 0

        # Check if device exists
        if device_num >= torch.cuda.device_count():
            logger.warning(f"CUDA device {device_num} not available, using device 0")
            device_num = 0

        logger.info(f"Using CUDA device {device_num}")
        return torch.device(f"cuda:{device_num}")

    @staticmethod
    def _get_mps_device() -> torch.device:
        """Get MPS device with fallback"""

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) device")
            return torch.device("mps")
        else:
            logger.warning("MPS not available, falling back to CPU")
            return torch.device("cpu")

    @staticmethod
    def _get_best_cuda_device() -> int:
        """Find CUDA device with most free memory"""

        if not torch.cuda.is_available():
            return 0

        max_free_memory = 0
        best_device = 0

        for i in range(torch.cuda.device_count()):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory

                # Set device to get current usage
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(i)
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory

                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = i

                # Restore original device
                torch.cuda.set_device(current_device)

            except Exception as e:
                logger.warning(f"Error checking CUDA device {i}: {e}")
                continue

        return best_device

    @staticmethod
    def get_device_info(device: torch.device) -> Dict[str, Any]:
        """
        Get detailed device information

        Args:
            device: torch.device object

        Returns:
            Dictionary with device information
        """
        info = {
            "device": str(device),
            "type": device.type,
            "platform": platform.system(),
        }

        if device.type == "cuda" and torch.cuda.is_available():
            try:
                device_idx = device.index if device.index is not None else 0
                props = torch.cuda.get_device_properties(device_idx)

                info.update(
                    {
                        "name": props.name,
                        "memory_total_gb": round(props.total_memory / (1024**3), 2),
                        "memory_allocated_gb": round(
                            torch.cuda.memory_allocated(device_idx) / (1024**3), 2
                        ),
                        "memory_cached_gb": round(
                            torch.cuda.memory_reserved(device_idx) / (1024**3), 2
                        ),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multiprocessor_count,
                    }
                )
            except Exception as e:
                logger.warning(f"Error getting CUDA device info: {e}")
                info["error"] = str(e)

        elif device.type == "mps":
            info.update(
                {"name": "Apple MPS", "description": "Apple Metal Performance Shaders"}
            )
        else:
            info.update(
                {
                    "name": "CPU",
                    "description": f"{platform.processor()} on {platform.system()}",
                }
            )

        return info
