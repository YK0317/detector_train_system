"""
Intelligent path resolution with cross-platform support
Fixes hard-coded path issues in the train system
"""

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class PathResolver:
    """Intelligent path resolution with cross-platform support"""

    @staticmethod
    def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Resolve paths with environment variables and relative paths

        Args:
            path_str: Path string to resolve
            base_dir: Base directory for relative paths

        Returns:
            Resolved Path object or None if invalid
        """
        if not path_str:
            return None

        try:
            # Handle environment variables
            path_str = os.path.expandvars(path_str)

            # Handle user home directory
            path_str = os.path.expanduser(path_str)

            path = Path(path_str)

            # Make relative paths relative to base_dir or current working directory
            if not path.is_absolute():
                base = base_dir or Path.cwd()
                path = base / path

            return path.resolve()
        except Exception as e:
            logger.warning(f"Failed to resolve path '{path_str}': {e}")
            return None

    @staticmethod
    def resolve_config_paths(
        config_dict: Dict[str, Any], base_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Recursively resolve all paths in a configuration dictionary

        Args:
            config_dict: Configuration dictionary
            base_dir: Base directory for relative paths

        Returns:
            Updated configuration with resolved paths
        """
        path_keys = {
            "path",
            "train_path",
            "val_path",
            "test_path",
            "output_dir",
            "real_path",
            "fake_path",
            "checkpoint_path",
            "data_path",
            "model_path",
            "config_path",
            "log_path",
        }

        def _resolve_recursive(obj, current_base):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if key in path_keys and isinstance(value, str):
                        resolved = PathResolver.resolve_path(value, current_base)
                        result[key] = str(resolved) if resolved else value
                    else:
                        result[key] = _resolve_recursive(value, current_base)
                return result
            elif isinstance(obj, list):
                return [_resolve_recursive(item, current_base) for item in obj]
            else:
                return obj

        return _resolve_recursive(config_dict, base_dir)

    @staticmethod
    def get_platform_paths() -> Dict[str, str]:
        """Get platform-specific default paths"""
        system = platform.system().lower()
        home = os.path.expanduser("~")

        if system == "windows":
            return {
                "home": home,
                "data": f"{home}\\Documents\\train_system_data",
                "models": f"{home}\\Documents\\train_system_models",
                "output": f"{home}\\Documents\\train_system_output",
                "cache": f"{home}\\AppData\\Local\\train_system\\cache",
            }
        elif system == "darwin":  # macOS
            return {
                "home": home,
                "data": f"{home}/Documents/train_system_data",
                "models": f"{home}/Documents/train_system_models",
                "output": f"{home}/Documents/train_system_output",
                "cache": f"{home}/Library/Caches/train_system",
            }
        else:  # Linux and others
            return {
                "home": home,
                "data": f"{home}/train_system_data",
                "models": f"{home}/train_system_models",
                "output": f"{home}/train_system_output",
                "cache": f"{home}/.cache/train_system",
            }

    @staticmethod
    def ensure_path_exists(path: Union[str, Path], create_parents: bool = True) -> bool:
        """
        Ensure a path exists, creating it if necessary

        Args:
            path: Path to check/create
            create_parents: Whether to create parent directories

        Returns:
            True if path exists or was created successfully
        """
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                if create_parents:
                    path_obj.mkdir(parents=True, exist_ok=True)
                else:
                    path_obj.mkdir(exist_ok=True)
                logger.info(f"Created directory: {path_obj}")
            return True
        except Exception as e:
            logger.error(f"Failed to create path {path}: {e}")
            return False
