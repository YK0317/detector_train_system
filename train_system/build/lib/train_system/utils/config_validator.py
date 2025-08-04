"""
Comprehensive configuration validation with auto-fixing
Fixes configuration issues and provides smart defaults
"""

import logging
import platform
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import torch

from .path_resolver import PathResolver
from .device_manager import DeviceManager

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Comprehensive configuration validation with auto-fixing"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.auto_fixes = []
        self.validation_results = {}
    
    def validate_and_fix(self, config_dict: Dict[str, Any], config_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate configuration and apply auto-fixes
        
        Args:
            config_dict: Configuration dictionary
            config_path: Path to configuration file (for relative path resolution)
            
        Returns:
            Fixed configuration dictionary
        """
        self.errors.clear()
        self.warnings.clear()
        self.auto_fixes.clear()
        self.validation_results.clear()
        
        logger.info("Starting configuration validation and auto-fixing...")
        
        # Determine base directory for path resolution
        base_dir = config_path.parent if config_path else Path.cwd()
        
        # Apply fixes
        config_dict = self._fix_paths(config_dict, base_dir)
        config_dict = self._fix_device_settings(config_dict)
        config_dict = self._fix_data_config(config_dict)
        config_dict = self._fix_model_config(config_dict)
        config_dict = self._fix_training_config(config_dict)
        config_dict = self._fix_output_config(config_dict)
        
        # Validate required fields
        self._validate_required_fields(config_dict)
        
        # Apply smart defaults
        config_dict = self._apply_smart_defaults(config_dict)
        
        # Log results
        self._log_validation_results()
        
        return config_dict
    
    def _fix_paths(self, config_dict: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """Fix and validate paths"""
        try:
            config_dict = PathResolver.resolve_config_paths(config_dict, base_dir)
            self.auto_fixes.append("Resolved relative paths to absolute paths")
        except Exception as e:
            self.warnings.append(f"Path resolution failed: {e}")
            
        return config_dict
    
    def _fix_device_settings(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fix device configuration"""
        device_str = config_dict.get('device', 'auto')
        
        try:
            device = DeviceManager.get_optimal_device(device_str)
            new_device_str = str(device)
            config_dict['device'] = new_device_str
            
            if device_str != new_device_str:
                self.auto_fixes.append(f"Changed device from '{device_str}' to '{new_device_str}' (availability check)")
                
        except Exception as e:
            config_dict['device'] = 'cpu'
            self.auto_fixes.append(f"Changed device to 'cpu' due to error: {e}")
            
        return config_dict
    
    def _fix_data_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fix data configuration issues"""
        data_config = config_dict.get('data', {})
        
        # Fix class mapping for binary classification
        if data_config.get('type') == 'class_folders':
            if 'class_mapping' not in data_config:
                class_folders = data_config.get('class_folders', {})
                
                if 'real_path' in data_config and 'fake_path' in data_config:
                    class_folders.update({'real': 'real', 'fake': 'fake'})
                    self.auto_fixes.append("Generated class_folders from real_path/fake_path")
                
                if class_folders:
                    data_config['class_mapping'] = {name: idx for idx, name in enumerate(class_folders.keys())}
                    self.auto_fixes.append("Auto-generated class mapping from class folders")
        
        # Fix batch size for device compatibility
        batch_size = data_config.get('batch_size', 32)
        device_str = config_dict.get('device', 'cpu')
        
        if device_str == 'cpu' and batch_size > 16:
            data_config['batch_size'] = 16
            self.auto_fixes.append(f"Reduced batch_size from {batch_size} to 16 for CPU compatibility")
        
        # Fix num_workers for Windows
        if platform.system() == 'Windows':
            num_workers = data_config.get('num_workers', 4)
            if num_workers > 2:
                data_config['num_workers'] = 2
                self.auto_fixes.append(f"Reduced num_workers from {num_workers} to 2 for Windows stability")
        
        return config_dict
    
    def _fix_model_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fix model configuration issues"""
        model_config = config_dict.get('model', {})
        
        # Ensure adapter is properly set
        if 'adapter' not in model_config or model_config['adapter'] is None:
            model_config['adapter'] = 'auto'
            self.auto_fixes.append("Set adapter to 'auto' for automatic detection")
        
        # Fix model args
        if 'num_classes' in model_config and 'model_args' in model_config:
            model_args = model_config['model_args']
            if 'num_classes' not in model_args and 'num_class' not in model_args:
                model_args['num_classes'] = model_config['num_classes']
                self.auto_fixes.append("Added num_classes to model_args")
        
        return config_dict
    
    def _fix_training_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fix training configuration issues"""
        training_config = config_dict.get('training', {})
        
        # Fix scheduler parameters
        scheduler = training_config.get('scheduler')
        scheduler_params = training_config.get('scheduler_params', {})
        
        if scheduler == 'cosine':
            epochs = training_config.get('epochs', 100)
            if 'T_max' not in scheduler_params:
                scheduler_params['T_max'] = epochs
                self.auto_fixes.append(f"Set T_max to {epochs} for cosine scheduler")
            elif scheduler_params['T_max'] != epochs:
                scheduler_params['T_max'] = epochs
                self.auto_fixes.append(f"Updated T_max to match epochs ({epochs}) for cosine scheduler")
        
        training_config['scheduler_params'] = scheduler_params
        
        return config_dict
    
    def _fix_output_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Fix output configuration issues"""
        output_config = config_dict.get('output', {})
        
        # Ensure output directory exists
        output_dir = output_config.get('output_dir', 'training_output')
        if PathResolver.ensure_path_exists(output_dir):
            self.auto_fixes.append(f"Ensured output directory exists: {output_dir}")
        
        return config_dict
    
    def _validate_required_fields(self, config_dict: Dict[str, Any]):
        """Validate that required fields are present"""
        required_sections = ['model', 'data', 'training']
        
        for section in required_sections:
            if section not in config_dict:
                self.errors.append(f"Missing required section: {section}")
    
    def _apply_smart_defaults(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply smart defaults for missing optional fields"""
        
        # Smart defaults for data configuration
        data_config = config_dict.setdefault('data', {})
        data_defaults = {
            'batch_size': 32,
            'num_workers': 4 if platform.system() != 'Windows' else 2,
            'pin_memory': config_dict.get('device', 'cpu') != 'cpu',
            'shuffle': True
        }
        
        for key, default_value in data_defaults.items():
            if key not in data_config:
                data_config[key] = default_value
                self.auto_fixes.append(f"Applied smart default: data.{key} = {default_value}")
        
        # Smart defaults for training configuration
        training_config = config_dict.setdefault('training', {})
        training_defaults = {
            'weight_decay': 0.0001,
            'save_frequency': 5,
            'val_frequency': 1,
            'early_stopping_patience': 10,
            'metrics_frequency': 100,
            'non_blocking_transfer': True,
            'efficient_validation': True
        }
        
        for key, default_value in training_defaults.items():
            if key not in training_config:
                training_config[key] = default_value
                self.auto_fixes.append(f"Applied smart default: training.{key} = {default_value}")
        
        # Handle resume from checkpoint
        if 'resume_from_checkpoint' in training_config:
            checkpoint_path = training_config['resume_from_checkpoint']
            if checkpoint_path:
                # Resolve checkpoint path
                resolved_path = PathResolver.resolve_path(checkpoint_path)
                if resolved_path and resolved_path.exists():
                    training_config['resume_from_checkpoint'] = str(resolved_path)
                    self.auto_fixes.append(f"Resolved checkpoint path: {checkpoint_path}")
                else:
                    self.warnings.append(f"Checkpoint file not found: {checkpoint_path}")
                    training_config['resume_from_checkpoint'] = None
        
        # Smart defaults for output configuration
        output_config = config_dict.setdefault('output', {})
        output_defaults = {
            'save_best_only': False,
            'save_lightweight': True
        }
        
        for key, default_value in output_defaults.items():
            if key not in output_config:
                output_config[key] = default_value
                self.auto_fixes.append(f"Applied smart default: output.{key} = {default_value}")
        
        return config_dict
    
    def _log_validation_results(self):
        """Log validation results"""
        if self.auto_fixes:
            logger.info("Configuration auto-fixes applied:")
            for fix in self.auto_fixes:
                logger.info(f"  ✓ {fix}")
        
        if self.warnings:
            logger.warning("Configuration warnings:")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if self.errors:
            logger.error("Configuration errors:")
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        
        if not self.errors and not self.warnings:
            logger.info("✅ Configuration validation completed successfully")
