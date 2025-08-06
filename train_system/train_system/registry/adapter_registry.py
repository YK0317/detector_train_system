"""
Adapter Registry for auto-discovery of custom model adapters.

Automatically scans specified directories for adapter classes and makes them
available for use without requiring manual file path configuration.
"""

import os
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Type, Optional, Set
import logging

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Auto-discovery registry for custom model adapters"""

    def __init__(self):
        self._adapters: Dict[str, Type] = {}
        self._adapter_info: Dict[str, dict] = {}
        self._search_paths = [
            "custom_adapters",
            "adapters",
            "train_system/adapters",
            "external_adapters",
            "model_adapters",
        ]
        self._base_class_names = {"ModelAdapter", "BaseAdapter", "Adapter"}
        self._scanned_paths: Set[str] = set()

    def scan_for_adapters(self, additional_paths: List[str] = None):
        """Scan for adapter classes in specified directories"""
        search_paths = self._search_paths.copy()
        if additional_paths:
            search_paths.extend(additional_paths)

        for search_path in search_paths:
            # Convert to absolute path and check if already scanned
            abs_path = os.path.abspath(search_path)
            if abs_path in self._scanned_paths:
                continue

            if os.path.exists(search_path):
                logger.debug(f"Scanning for adapters in: {search_path}")
                self._scan_directory(search_path)
                self._scanned_paths.add(abs_path)
            else:
                logger.debug(f"Adapter search path does not exist: {search_path}")

        logger.info(
            f"Found {len(self._adapters)} adapters: {list(self._adapters.keys())}"
        )

    def _scan_directory(self, directory: str):
        """Scan a directory for adapter files"""
        try:
            for file_path in Path(directory).rglob("*.py"):
                if file_path.name.startswith("_") or file_path.name == "__init__.py":
                    continue

                try:
                    self._load_adapters_from_file(file_path)
                except Exception as e:
                    logger.debug(f"Could not load adapters from {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {e}")

    def _load_adapters_from_file(self, file_path: Path):
        """Load adapter classes from a Python file"""
        try:
            # Create unique module name
            module_name = f"adapter_registry_{file_path.stem}_{id(file_path)}"

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

            # Find adapter classes
            found_adapters = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_adapter_class(obj) and not name.startswith("_"):
                    adapter_name = self._generate_adapter_name(name, file_path)

                    # Avoid duplicates
                    if adapter_name not in self._adapters:
                        self._adapters[adapter_name] = obj
                        self._adapter_info[adapter_name] = {
                            "class_name": name,
                            "file_path": str(file_path),
                            "module_name": module_name,
                            "docstring": obj.__doc__ or "No description available",
                        }
                        found_adapters.append(adapter_name)
                        logger.debug(f"Registered adapter: {adapter_name} -> {obj}")

            if not found_adapters:
                # Clean up if no adapters found
                if module_name in sys.modules:
                    del sys.modules[module_name]

        except Exception as e:
            logger.debug(f"Error loading adapters from {file_path}: {e}")

    def _is_adapter_class(self, cls) -> bool:
        """Check if class is an adapter"""
        if not inspect.isclass(cls):
            return False

        # Skip if it's from another module that we know is not an adapter
        module_name = getattr(cls, "__module__", "")
        if any(skip in module_name for skip in ["torch.", "numpy.", "builtins"]):
            return False

        # Check inheritance for base adapter classes
        for base in inspect.getmro(cls):
            if base.__name__ in self._base_class_names:
                return True

        # Check interface (duck typing) - look for essential adapter methods
        required_methods = ["extract_logits", "get_predictions"]
        optional_methods = ["extract_features", "__call__"]

        has_required = all(
            hasattr(cls, method) and callable(getattr(cls, method))
            for method in required_methods
        )
        has_optional = any(
            hasattr(cls, method) and callable(getattr(cls, method))
            for method in optional_methods
        )

        # Must have required methods and at least one optional method
        return has_required and has_optional

    def _generate_adapter_name(self, class_name: str, file_path: Path) -> str:
        """Generate a friendly name for the adapter"""
        # Remove common suffixes
        name = class_name.lower()
        for suffix in ["adapter", "wrapper", "handler", "modeladapter"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break

        # Use filename if class name is generic
        if name in ["custom", "external", "base", "model"]:
            name = file_path.stem.lower()
            # Remove common file prefixes/suffixes
            for fix in ["adapter_", "_adapter", "model_", "_model"]:
                if fix.startswith("_"):
                    if name.endswith(fix):
                        name = name[: -len(fix)]
                else:
                    if name.startswith(fix):
                        name = name[len(fix) :]

        # Ensure name is valid
        if not name or name in ["custom", "base"]:
            name = file_path.stem.lower()

        return name

    def get_adapter(self, name: str) -> Optional[Type]:
        """Get adapter class by name"""
        return self._adapters.get(name.lower())

    def get_adapter_info(self, name: str) -> Optional[dict]:
        """Get detailed information about an adapter"""
        return self._adapter_info.get(name.lower())

    def list_adapters(self) -> Dict[str, Type]:
        """List all registered adapters"""
        return self._adapters.copy()

    def list_adapter_info(self) -> Dict[str, dict]:
        """List detailed information about all adapters"""
        return self._adapter_info.copy()

    def register_adapter(self, name: str, adapter_class: Type, force: bool = False):
        """Manually register an adapter"""
        name = name.lower()
        if name in self._adapters and not force:
            logger.warning(
                f"Adapter '{name}' already registered. Use force=True to override."
            )
            return

        self._adapters[name] = adapter_class
        self._adapter_info[name] = {
            "class_name": adapter_class.__name__,
            "file_path": "manually_registered",
            "module_name": adapter_class.__module__,
            "docstring": adapter_class.__doc__ or "Manually registered adapter",
        }
        logger.info(f"Manually registered adapter: {name}")

    def unregister_adapter(self, name: str):
        """Remove an adapter from registry"""
        name = name.lower()
        if name in self._adapters:
            del self._adapters[name]
            if name in self._adapter_info:
                del self._adapter_info[name]
            logger.info(f"Unregistered adapter: {name}")
        else:
            logger.warning(f"Adapter '{name}' not found in registry")

    def clear(self):
        """Clear all registered adapters"""
        self._adapters.clear()
        self._adapter_info.clear()
        self._scanned_paths.clear()
        logger.info("Cleared adapter registry")


# Global registry instance
adapter_registry = AdapterRegistry()
