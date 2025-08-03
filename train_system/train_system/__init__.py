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
try:
    from .core.wrapper import UnifiedTrainingWrapper, ModelFactory
    from .core.trainer import UnifiedTrainer
    from .core.dataset import UnifiedDataset
    
    # Adapter imports
    from .adapters import (
        ModelAdapter,
        StandardAdapter,
        LogitsAndFeaturesAdapter,
        DictOutputAdapter,
        CapsuleNetworkAdapter,
        CustomFunctionAdapter,
        AutoAdapter,
        get_adapter_for_model
    )
    
    # Configuration imports
    from .config import (
        ConfigValidator,
        ValidationResult,
        ConfigTemplateManager,
        UnifiedTrainingConfig,
        ModelConfig,
        DataConfig,
        TrainingConfig
    )
    
    # API imports
    from .api.server import TrainingAPI
    
    # Registry imports
    from .registry import (
        AdapterRegistry,
        TrainerRegistry,
        initialize_registries,
        list_available_components,
        scan_additional_paths,
        get_component_by_name,
        register_component,
        is_component_available
    )
    
except ImportError as e:
    # Graceful handling of import errors during development
    import warnings
    warnings.warn(f"Some train_system components could not be imported: {e}")

__all__ = [
    # Core classes
    "UnifiedTrainingWrapper",
    "ModelFactory", 
    "UnifiedTrainer",
    "UnifiedDataset",
    
    # Adapters
    "ModelAdapter",
    "StandardAdapter",
    "LogitsAndFeaturesAdapter", 
    "DictOutputAdapter",
    "CapsuleNetworkAdapter",
    "CustomFunctionAdapter",
    "AutoAdapter",
    "get_adapter_for_model",
    
    # Configuration
    "ConfigValidator",
    "ValidationResult",
    "ConfigTemplateManager",
    "UnifiedTrainingConfig",
    "ModelConfig",
    "DataConfig", 
    "TrainingConfig",
    
    # API
    "TrainingAPI",
    
    # Registry
    "AdapterRegistry",
    "TrainerRegistry", 
    "initialize_registries",
    "list_available_components",
    "scan_additional_paths",
    "get_component_by_name",
    "register_component",
    "is_component_available",
]
