"""
External Adapter Loader

This module provides functionality to load and use custom adapters from external files.
Users can provide their own adapter implementations without modifying the core codebase.
"""

import importlib.util
import sys
from pathlib import Path
import logging

# Import base adapter to check inheritance
from . import ModelAdapter

logger = logging.getLogger(__name__)


def load_external_adapter(script_path, class_name, **kwargs):
    """
    Dynamically load an external adapter class from a script

    Args:
        script_path (str): Path to the script containing the adapter class
        class_name (str): Name of the adapter class
        **kwargs: Additional arguments for the adapter

    Returns:
        class: The loaded adapter class
    """
    script_path = Path(script_path)

    if not script_path.exists():
        raise FileNotFoundError(f"External adapter script not found: {script_path}")

    # Load the module
    module_name = f"external_adapter_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None:
        raise ImportError(f"Could not load spec for module: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the adapter class
    if not hasattr(module, class_name):
        raise AttributeError(f"Adapter class '{class_name}' not found in {script_path}")

    adapter_class = getattr(module, class_name)

    # Check if it's a valid adapter class (inherit from ModelAdapter)
    if not issubclass(adapter_class, ModelAdapter):
        logger.warning(
            f"External adapter class {class_name} doesn't inherit from ModelAdapter. "
            + "This might cause unexpected behavior."
        )

    return adapter_class


class ExternalAdapterLoader:
    """Handles loading and initialization of external adapters"""

    @staticmethod
    def get_adapter(config, model):
        """
        Load and initialize an external adapter based on configuration

        Args:
            config: Configuration object with external_adapter field
            model: Model to adapt

        Returns:
            ModelAdapter: Initialized adapter instance
        """
        if (
            not hasattr(config.model, "external_adapter")
            or not config.model.external_adapter
        ):
            raise ValueError("No external adapter configuration found")

        ext_adapter_config = config.model.external_adapter

        # Get required fields
        script_path = ext_adapter_config.get("script_path")
        class_name = ext_adapter_config.get("class_name")

        if not script_path or not class_name:
            raise ValueError(
                "External adapter config must include script_path and class_name"
            )

        # Get adapter parameters
        adapter_params = ext_adapter_config.get("parameters", {})

        # Load adapter class
        adapter_class = load_external_adapter(script_path, class_name)

        # Create adapter instance
        adapter = adapter_class(model, **adapter_params)

        logger.info(f"Loaded external adapter: {class_name} from {script_path}")

        return adapter
