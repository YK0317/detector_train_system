"""
Discovery and initialization functions for the registry system.

Provides high-level functions to initialize registries and list available components.
"""

from .adapter_registry import adapter_registry
from .trainer_registry import trainer_registry
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def initialize_registries(config=None, force_rescan: bool = False, additional_paths: dict = None):
    """
    Initialize both registries by scanning for custom components.
    
    Args:
        config: Configuration object that may contain registry settings
        force_rescan: Whether to force a complete rescan even if already scanned
        additional_paths: Dict with 'adapter_paths' and 'trainer_paths' lists
    """
    logger.info("Initializing component registries...")
    
    # Clear registries if force rescan is requested
    if force_rescan:
        adapter_registry.clear()
        trainer_registry.clear()
    
    # Get additional search paths from various sources
    additional_adapter_paths = []
    additional_trainer_paths = []
    
    # From config object
    if config and hasattr(config, 'registry'):
        additional_adapter_paths.extend(getattr(config.registry, 'adapter_paths', []))
        additional_trainer_paths.extend(getattr(config.registry, 'trainer_paths', []))
    
    # From direct parameter
    if additional_paths:
        additional_adapter_paths.extend(additional_paths.get('adapter_paths', []))
        additional_trainer_paths.extend(additional_paths.get('trainer_paths', []))
    
    try:
        # Scan for components
        adapter_registry.scan_for_adapters(additional_adapter_paths)
        trainer_registry.scan_for_trainers(additional_trainer_paths)
        
        logger.info("Registry initialization complete")
        
    except Exception as e:
        logger.error(f"Error during registry initialization: {e}")
        raise


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
                if info.get('docstring') != "No description available":
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
                if info.get('docstring') != "No description available":
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
    
    return {'adapters': list(adapters.keys()), 'trainers': list(trainers.keys())}


def scan_additional_paths(adapter_paths: List[str] = None, trainer_paths: List[str] = None):
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
    if component_type.lower() == 'adapter':
        return adapter_registry.get_adapter(name)
    elif component_type.lower() == 'trainer':
        return trainer_registry.get_trainer(name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def register_component(component_type: str, name: str, component_class, force: bool = False):
    """
    Manually register a component.
    
    Args:
        component_type: 'adapter' or 'trainer'
        name: Name to register the component under
        component_class: The component class
        force: Whether to override existing registration
    """
    if component_type.lower() == 'adapter':
        adapter_registry.register_adapter(name, component_class, force)
    elif component_type.lower() == 'trainer':
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
