"""
Discovery and initialization functions for the registry system.

Provides high-level functions to initialize registries and list available components.
Uses dynamic discovery to replace hard-coded paths.
"""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.path_resolver import PathResolver
from .adapter_registry import adapter_registry
from .trainer_registry import trainer_registry

logger = logging.getLogger(__name__)


def initialize_registries(
    config=None, force_rescan: bool = False, additional_paths: dict = None
):
    """
    Initialize both registries by scanning for custom components using dynamic discovery.

    Args:
        config: Configuration object that may contain registry settings
        force_rescan: Whether to force a complete rescan even if already scanned
        additional_paths: Dict with 'adapter_paths' and 'trainer_paths' lists
    """
    logger.info("Initializing component registries with dynamic discovery...")

    # Clear registries if force rescan is requested
    if force_rescan:
        adapter_registry.clear()
        trainer_registry.clear()

    # Get dynamic search paths
    search_paths = _get_dynamic_search_paths(config, additional_paths)

    try:
        # Use dynamic discovery to find components
        discovered_components = _discover_components_dynamically(search_paths)

        # Register discovered adapters
        for name, component_info in discovered_components.get("adapters", {}).items():
            adapter_registry.register_adapter(name, component_info["class"])
            logger.debug(f"Registered adapter: {name}")

        # Register discovered trainers
        for name, component_info in discovered_components.get("trainers", {}).items():
            trainer_registry.register_trainer(name, component_info["class"])
            logger.debug(f"Registered trainer: {name}")

        adapter_count = len(discovered_components.get("adapters", {}))
        trainer_count = len(discovered_components.get("trainers", {}))

        logger.info(
            f"Found {adapter_count} adapters: {list(discovered_components.get('adapters', {}).keys())}"
        )
        logger.info(
            f"Found {trainer_count} trainers: {list(discovered_components.get('trainers', {}).keys())}"
        )
        logger.info("Registry initialization complete")

    except Exception as e:
        logger.error(f"Registry initialization failed: {e}")
        raise


def _get_dynamic_search_paths(config=None, additional_paths: dict = None) -> List[str]:
    """Get intelligent search paths for component discovery"""

    # Start with platform-specific paths
    platform_paths = PathResolver.get_platform_paths()

    search_paths = [
        # Current directory patterns
        "./custom_adapters",
        "./custom_trainers",
        "./adapters",
        "./trainers",
        "./components",
        # User-specific paths
        f"{platform_paths['home']}/train_system/adapters",
        f"{platform_paths['home']}/train_system/trainers",
        f"{platform_paths['home']}/train_system/components",
    ]

    # Add package internal paths
    try:
        package_root = Path(__file__).parent.parent.parent
        package_paths = [
            str(package_root / "custom_adapters"),
            str(package_root / "custom_trainers"),
            str(package_root / "adapters"),
            str(package_root / "trainers"),
        ]
        search_paths.extend(package_paths)
    except Exception as e:
        logger.debug(f"Could not determine package paths: {e}")

    # Add paths from config
    if config and hasattr(config, "registry"):
        config_paths = []
        if hasattr(config.registry, "adapter_paths"):
            config_paths.extend(config.registry.adapter_paths)
        if hasattr(config.registry, "trainer_paths"):
            config_paths.extend(config.registry.trainer_paths)
        search_paths.extend(config_paths)

    # Add additional paths
    if additional_paths:
        search_paths.extend(additional_paths.get("adapter_paths", []))
        search_paths.extend(additional_paths.get("trainer_paths", []))

    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in search_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    return unique_paths


def _discover_components_dynamically(
    search_paths: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Dynamically discover components from search paths"""

    results = {"adapters": {}, "trainers": {}, "errors": []}

    for search_path in search_paths:
        try:
            path = Path(search_path)
            if path.exists() and path.is_dir():
                _scan_directory_for_components(path, results)
            elif path.exists() and path.is_file() and path.suffix == ".py":
                _scan_file_for_components(path, results)
        except Exception as e:
            error_msg = f"Error scanning {search_path}: {e}"
            results["errors"].append(error_msg)
            logger.debug(error_msg)

    return results


def _scan_directory_for_components(directory: Path, results: Dict[str, Dict[str, Any]]):
    """Scan a directory for component files"""

    for py_file in directory.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
        _scan_file_for_components(py_file, results)


def _scan_file_for_components(py_file: Path, results: Dict[str, Dict[str, Any]]):
    """Scan a Python file for adapter and trainer classes"""

    try:
        # Load module dynamically
        module_name = f"dynamic_module_{py_file.stem}_{id(py_file)}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)

        # Execute module in isolated environment
        old_modules = sys.modules.copy()
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Look for component classes
            for name in dir(module):
                if name.startswith("_"):
                    continue

                obj = getattr(module, name)
                if not inspect.isclass(obj):
                    continue

                # Check if it's an adapter
                if _is_adapter_class(obj):
                    adapter_name = _get_component_name(obj.__name__, "Adapter")
                    results["adapters"][adapter_name] = {
                        "class": obj,
                        "module": module_name,
                        "file": str(py_file),
                    }

                # Check if it's a trainer
                elif _is_trainer_class(obj):
                    trainer_name = _get_component_name(obj.__name__, "Trainer")
                    results["trainers"][trainer_name] = {
                        "class": obj,
                        "module": module_name,
                        "file": str(py_file),
                    }

        finally:
            # Clean up module imports
            for module_key in list(sys.modules.keys()):
                if module_key not in old_modules:
                    del sys.modules[module_key]

    except Exception as e:
        error_msg = f"Error loading {py_file}: {e}"
        results["errors"].append(error_msg)
        logger.debug(error_msg)


def _is_adapter_class(cls) -> bool:
    """Check if class is an adapter"""
    # Check by naming convention
    if cls.__name__.endswith("Adapter"):
        return True

    # Check by interface
    required_methods = ["process_outputs"]
    return all(hasattr(cls, method) for method in required_methods)


def _is_trainer_class(cls) -> bool:
    """Check if class is a trainer"""
    # Check by naming convention
    if cls.__name__.endswith("Trainer"):
        return True

    # Check by interface
    required_methods = ["train"]
    return all(hasattr(cls, method) for method in required_methods)


def _get_component_name(class_name: str, suffix: str) -> str:
    """Extract component name from class name"""
    if suffix and class_name.endswith(suffix):
        name = class_name[: -len(suffix)]
    else:
        name = class_name

    # Convert CamelCase to lowercase with underscores
    import re

    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return name


def list_available_components(verbose: bool = False):
    """
    List all available adapters and trainers.

    Args:
        verbose: Whether to show detailed information about each component
    """
    adapters = adapter_registry.list_adapters()
    trainers = trainer_registry.list_trainers()

    print("\n🎯 Available Components:")
    print("=" * 50)

    print(f"\n📋 Adapters ({len(adapters)}):")
    if not adapters:
        print("   No adapters found.")
    else:
        for name, cls in adapters.items():
            if verbose:
                info = adapter_registry.get_adapter_info(name)
                print(f"   • {name}")
                print(f"     Class: {cls.__name__}")
                print(f"     File: {info.get('file_path', 'unknown')}")
                if info.get("docstring") != "No description available":
                    print(f"     Description: {info.get('docstring', '')[:80]}...")
            else:
                print(f"   • {name} -> {cls.__name__}")

    print(f"\n🏃 Trainers ({len(trainers)}):")
    if not trainers:
        print("   No trainers found.")
    else:
        for name, cls in trainers.items():
            if verbose:
                info = trainer_registry.get_trainer_info(name)
                print(f"   • {name}")
                print(f"     Class: {cls.__name__}")
                print(f"     File: {info.get('file_path', 'unknown')}")
                if info.get("docstring") != "No description available":
                    print(f"     Description: {info.get('docstring', '')[:80]}...")
            else:
                print(f"   • {name} -> {cls.__name__}")

    if adapters or trainers:
        print("\n💡 Usage:")
        if adapters:
            print("   model:")
            print("     adapter: 'adapter_name'")
        if trainers:
            print("   external_trainer:")
            print("     name: 'trainer_name'")

    return {"adapters": list(adapters.keys()), "trainers": list(trainers.keys())}


def scan_additional_paths(
    adapter_paths: List[str] = None, trainer_paths: List[str] = None
):
    """
    Scan additional paths for components.

    Args:
        adapter_paths: Additional paths to scan for adapters
        trainer_paths: Additional paths to scan for trainers
    """
    if adapter_paths:
        logger.info(f"Scanning additional adapter paths: {adapter_paths}")
        adapter_registry.scan_for_adapters(adapter_paths)

    if trainer_paths:
        logger.info(f"Scanning additional trainer paths: {trainer_paths}")
        trainer_registry.scan_for_trainers(trainer_paths)


def get_component_by_name(component_type: str, name: str):
    """
    Get a component by type and name.

    Args:
        component_type: 'adapter' or 'trainer'
        name: Name of the component

    Returns:
        Component class or None if not found
    """
    if component_type.lower() == "adapter":
        return adapter_registry.get_adapter(name)
    elif component_type.lower() == "trainer":
        return trainer_registry.get_trainer(name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def register_component(
    component_type: str, name: str, component_class, force: bool = False
):
    """
    Manually register a component.

    Args:
        component_type: 'adapter' or 'trainer'
        name: Name to register the component under
        component_class: The component class
        force: Whether to override existing registration
    """
    if component_type.lower() == "adapter":
        adapter_registry.register_adapter(name, component_class, force)
    elif component_type.lower() == "trainer":
        trainer_registry.register_trainer(name, component_class, force)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def is_component_available(component_type: str, name: str) -> bool:
    """
    Check if a component is available.

    Args:
        component_type: 'adapter' or 'trainer'
        name: Name of the component

    Returns:
        True if component is available, False otherwise
    """
    component = get_component_by_name(component_type, name)
    return component is not None
