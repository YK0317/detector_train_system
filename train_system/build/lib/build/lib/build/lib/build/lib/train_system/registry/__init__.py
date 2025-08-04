"""
Registry system for auto-discovery of custom adapters and trainers.

This module provides automatic discovery and registration of custom components,
making it easier for users to extend the train system without manual configuration.
"""

from .adapter_registry import AdapterRegistry, adapter_registry
from .trainer_registry import TrainerRegistry, trainer_registry
from .discovery import (
    initialize_registries, 
    list_available_components,
    scan_additional_paths,
    get_component_by_name,
    register_component,
    is_component_available
)

__all__ = [
    'AdapterRegistry', 'adapter_registry',
    'TrainerRegistry', 'trainer_registry', 
    'initialize_registries', 'list_available_components',
    'scan_additional_paths', 'get_component_by_name',
    'register_component', 'is_component_available'
]
