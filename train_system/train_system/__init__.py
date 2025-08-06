"""
Train System

A comprehensive training system that can train any model with any dataset
using configuration files and providing API access.

This package provides:
- Universal model wrapper for any PyTorch model
- Flexible adapter system for different model outputs
- Configuration-based training pipeline
- REST API for remote training control
- Command-line interface for easy usage

Main Components:
    - train_system.core: Core training functionality
    - train_system.adapters: Model output adapters
    - train_system.config: Configuration management
    - train_system.api: REST API server
    - train_system.cli: Command-line interface
    - train_system.utils: Utility functions

Example Usage:
    >>> from train_system import UnifiedTrainingWrapper, ModelFactory
    >>> from train_system.adapters import AutoAdapter
    >>>
    >>> # Create wrapped model
    >>> wrapper = ModelFactory.create_wrapped_model('resnet18')
    >>>
    >>> # Or wrap manually
    >>> wrapper = UnifiedTrainingWrapper(your_model, AutoAdapter())
    >>>
    >>> # Train normally
    >>> optimizer = torch.optim.Adam(wrapper.parameters())
    >>> for images, labels in dataloader:
    ...     outputs = wrapper(images)  # Returns only logits
    ...     loss = wrapper.compute_loss(outputs, labels)
    ...     loss.backward()
    ...     optimizer.step()
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "assistant@example.com"

# Core imports
_imported_components = []

try:
    # Configuration imports (minimal dependencies)
    from .config import (
        ConfigTemplateManager,
        ConfigValidator,
        DataConfig,
        ModelConfig,
        TrainingConfig,
        UnifiedTrainingConfig,
        ValidationResult,
    )

    _imported_components.extend(
        [
            "ConfigValidator",
            "ValidationResult",
            "ConfigTemplateManager",
            "UnifiedTrainingConfig",
            "ModelConfig",
            "DataConfig",
            "TrainingConfig",
        ]
    )
except ImportError as e:
    import warnings

    warnings.warn(f"Config components could not be imported: {e}")

try:
    # Adapter imports (require torch)
    from .adapters import (
        AutoAdapter,
        CapsuleNetworkAdapter,
        CustomFunctionAdapter,
        DictOutputAdapter,
        LogitsAndFeaturesAdapter,
        ModelAdapter,
        StandardAdapter,
        get_adapter_for_model,
    )

    _imported_components.extend(
        [
            "ModelAdapter",
            "StandardAdapter",
            "LogitsAndFeaturesAdapter",
            "DictOutputAdapter",
            "CapsuleNetworkAdapter",
            "CustomFunctionAdapter",
            "AutoAdapter",
            "get_adapter_for_model",
        ]
    )
except ImportError as e:
    import warnings

    warnings.warn(f"Adapter components could not be imported: {e}")

try:
    # Core training components (require torch)
    from .core.dataset import UnifiedDataset
    from .core.trainer import UnifiedTrainer
    from .core.wrapper import ModelFactory, UnifiedTrainingWrapper

    _imported_components.extend(
        ["UnifiedTrainingWrapper", "ModelFactory", "UnifiedTrainer", "UnifiedDataset"]
    )
except ImportError as e:
    import warnings

    warnings.warn(f"Core training components could not be imported: {e}")

try:
    # Registry imports
    from .registry import (
        AdapterRegistry,
        TrainerRegistry,
        get_component_by_name,
        initialize_registries,
        is_component_available,
        list_available_components,
        register_component,
        scan_additional_paths,
    )

    _imported_components.extend(
        [
            "AdapterRegistry",
            "TrainerRegistry",
            "initialize_registries",
            "list_available_components",
            "scan_additional_paths",
            "get_component_by_name",
            "register_component",
            "is_component_available",
        ]
    )
except ImportError as e:
    import warnings

    warnings.warn(f"Registry components could not be imported: {e}")

try:
    # API imports (require Flask)
    from .api.server import TrainingAPI

    _imported_components.append("TrainingAPI")
except ImportError as e:
    import warnings

    warnings.warn(
        f"API components could not be imported (Flask may not be available): {e}"
    )

# Dynamic __all__ based on successfully imported components
__all__ = _imported_components
