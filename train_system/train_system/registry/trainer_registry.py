"""
Trainer Registry for auto-discovery of external trainers.

Automatically scans specified directories for trainer classes and makes them
available for use without requiring manual file path configuration.
"""

import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class TrainerRegistry:
    """Auto-discovery registry for external trainers"""

    def __init__(self):
        self._trainers: Dict[str, Type] = {}
        self._trainer_info: Dict[str, dict] = {}
        self._search_paths = [
            "custom_trainers",
            "external_trainers",
            "trainers",
            "train_system/external_trainers",
            "train_system/custom_trainers",
        ]
        self._scanned_paths: Set[str] = set()

    def scan_for_trainers(self, additional_paths: List[str] = None):
        """Scan for trainer classes in specified directories"""
        search_paths = self._search_paths.copy()
        if additional_paths:
            search_paths.extend(additional_paths)

        for search_path in search_paths:
            # Convert to absolute path and check if already scanned
            abs_path = os.path.abspath(search_path)
            if abs_path in self._scanned_paths:
                continue

            if os.path.exists(search_path):
                logger.debug(f"Scanning for trainers in: {search_path}")
                self._scan_directory(search_path)
                self._scanned_paths.add(abs_path)
            else:
                logger.debug(f"Trainer search path does not exist: {search_path}")

        logger.info(
            f"Found {len(self._trainers)} trainers: {list(self._trainers.keys())}"
        )

    def _scan_directory(self, directory: str):
        """Scan a directory for trainer files"""
        try:
            for file_path in Path(directory).rglob("*.py"):
                if file_path.name.startswith("_") or file_path.name == "__init__.py":
                    continue

                try:
                    self._load_trainers_from_file(file_path)
                except Exception as e:
                    logger.debug(f"Could not load trainers from {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {e}")

    def _load_trainers_from_file(self, file_path: Path):
        """Load trainer classes from a Python file"""
        try:
            # Create unique module name
            module_name = f"trainer_registry_{file_path.stem}_{id(file_path)}"

            # Load module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.debug(f"Could not create spec for {file_path}")
                return

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules to handle relative imports
            sys.modules[module_name] = module

            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.debug(f"Error executing module {file_path}: {e}")
                if module_name in sys.modules:
                    del sys.modules[module_name]
                return

            # Find trainer classes
            found_trainers = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_trainer_class(obj) and not name.startswith("_"):
                    trainer_name = self._generate_trainer_name(name, file_path)

                    # Avoid duplicates
                    if trainer_name not in self._trainers:
                        self._trainers[trainer_name] = obj
                        self._trainer_info[trainer_name] = {
                            "class_name": name,
                            "file_path": str(file_path),
                            "module_name": module_name,
                            "docstring": obj.__doc__ or "No description available",
                        }
                        found_trainers.append(trainer_name)
                        logger.debug(f"Registered trainer: {trainer_name} -> {obj}")

            if not found_trainers:
                # Clean up if no trainers found
                if module_name in sys.modules:
                    del sys.modules[module_name]

        except Exception as e:
            logger.debug(f"Error loading trainers from {file_path}: {e}")

    def _is_trainer_class(self, cls) -> bool:
        """Check if class is a trainer"""
        if not inspect.isclass(cls):
            return False

        # Skip if it's from another module that we know is not a trainer
        module_name = getattr(cls, "__module__", "")
        if any(
            skip in module_name
            for skip in ["torch.", "numpy.", "builtins", "ultralytics"]
        ):
            return False

        # Skip specific classes that are not trainers
        class_name = cls.__name__
        if class_name in ["YOLO"]:  # ultralytics YOLO model class
            return False

        # Check for required trainer methods
        required_methods = ["train"]
        if not all(
            hasattr(cls, method) and callable(getattr(cls, method))
            for method in required_methods
        ):
            return False

        # Check constructor signature (should accept reasonable parameters)
        try:
            init_signature = inspect.signature(cls.__init__)
            init_params = list(init_signature.parameters.keys())[1:]  # Skip 'self'

            # Should have at least 2-3 parameters (typical: config, model, etc.)
            if len(init_params) < 2:
                return False

            # Look for common trainer parameter names
            common_params = {
                "config",
                "model",
                "train_loader",
                "val_loader",
                "optimizer",
                "criterion",
                "device",
                "logger",
            }
            param_matches = sum(
                1
                for param in init_params
                if any(common in param.lower() for common in common_params)
            )

            # Should have at least one parameter that looks trainer-like
            return param_matches >= 1

        except (ValueError, TypeError):
            # If we can't inspect the signature, be conservative
            return False

    def _generate_trainer_name(self, class_name: str, file_path: Path) -> str:
        """Generate a friendly name for the trainer"""
        # Remove common suffixes
        name = class_name.lower()
        for suffix in ["trainer", "training", "system", "builtin"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        # Use filename if class name is generic
        if name in ["external", "custom", "base"]:
            name = file_path.stem.lower()
            # Remove common file prefixes/suffixes
            for fix in ["trainer_", "_trainer", "external_", "_external"]:
                if fix.startswith("_"):
                    if name.endswith(fix):
                        name = name[: -len(fix)]
                else:
                    if name.startswith(fix):
                        name = name[len(fix) :]

        # Ensure name is valid
        if not name or name in ["custom", "base", "external"]:
            name = file_path.stem.lower()

        return name

    def get_trainer(self, name: str) -> Optional[Type]:
        """Get trainer class by name"""
        return self._trainers.get(name.lower())

    def get_trainer_info(self, name: str) -> Optional[dict]:
        """Get detailed information about a trainer"""
        return self._trainer_info.get(name.lower())

    def list_trainers(self) -> Dict[str, Type]:
        """List all registered trainers"""
        return self._trainers.copy()

    def list_trainer_info(self) -> Dict[str, dict]:
        """List detailed information about all trainers"""
        return self._trainer_info.copy()

    def register_trainer(self, name: str, trainer_class: Type, force: bool = False):
        """Manually register a trainer"""
        name = name.lower()
        if name in self._trainers and not force:
            logger.warning(
                f"Trainer '{name}' already registered. Use force=True to override."
            )
            return

        self._trainers[name] = trainer_class
        self._trainer_info[name] = {
            "class_name": trainer_class.__name__,
            "file_path": "manually_registered",
            "module_name": trainer_class.__module__,
            "docstring": trainer_class.__doc__ or "Manually registered trainer",
        }
        logger.info(f"Manually registered trainer: {name}")

    def unregister_trainer(self, name: str):
        """Remove a trainer from registry"""
        name = name.lower()
        if name in self._trainers:
            del self._trainers[name]
            if name in self._trainer_info:
                del self._trainer_info[name]
            logger.info(f"Unregistered trainer: {name}")
        else:
            logger.warning(f"Trainer '{name}' not found in registry")

    def clear(self):
        """Clear all registered trainers"""
        self._trainers.clear()
        self._trainer_info.clear()
        self._scanned_paths.clear()
        logger.info("Cleared trainer registry")


# Global registry instance
trainer_registry = TrainerRegistry()
