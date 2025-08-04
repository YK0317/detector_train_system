"""
Unified Trainer for Train System

Main training component that orchestrates the entire training process.
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from datetime import datetime
from datetime import datetime
import random
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json
import time
from tqdm import tqdm

from ..config import UnifiedTrainingConfig, ModelConfig
from ..core.wrapper import ModelFactory
from ..core.dataset import UnifiedDataset
from ..core.external_trainer import HybridTrainer
from ..utils.memory import optimize_memory, log_memory_usage, move_data_to_device, setup_cuda_optimizations, MemoryTracker
from ..utils.device_manager import DeviceManager
from ..utils.path_resolver import PathResolver
from ..utils.config_validator import ConfigValidator


class UnifiedTrainer:
    """
    Unified trainer that can train any model with any dataset
    Supports both built-in and external training methods
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        """
        Initialize unified trainer
        
        Args:
            config: Complete training configuration
        """
        # Setup core components first
        self.logger = self._setup_logging()
        
        # Validate and fix configuration
        self.config = self._validate_config(config)
        
        # Setup device after config validation
        self.device = self._setup_device()
        
        # Setup output directory after config is complete
        self._setup_output_directory()
        
        # Initialize registries for auto-discovery
        self._initialize_registries()
        
        # Check if external trainer should be used
        self.use_external_trainer = self._should_use_external_trainer()
        
        # Training state
        self.model = None
        self.wrapper = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Pretrained weights storage
        self._stored_optimizer_state = None
        self._stored_scheduler_state = None
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        self.current_epoch = 0
        
        # Performance optimizations (with safe defaults)
        self.memory_tracker = MemoryTracker()
        self.metrics_frequency = getattr(self.config.training, 'metrics_frequency', 100)
        self.non_blocking_transfer = getattr(self.config.training, 'non_blocking_transfer', True)
        self.efficient_validation = getattr(self.config.training, 'efficient_validation', True)
        
        # Setup reproducibility
        self._setup_reproducibility()
        
        # Setup CUDA optimizations
        setup_cuda_optimizations()
    
    def _validate_config(self, config: UnifiedTrainingConfig) -> UnifiedTrainingConfig:
        """
        Validate and fix configuration using ConfigValidator
        
        Args:
            config: Original configuration
            
        Returns:
            Validated and fixed configuration
        """
        # Convert config to dict for validation
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            # If config doesn't have to_dict, create dict from attributes
            config_dict = self._config_to_dict(config)
        
        # Validate and fix
        validator = ConfigValidator()
        fixed_config_dict = validator.validate_and_fix(config_dict)
        
        # Update original config with fixes while preserving structure
        self._update_config_from_dict(config, fixed_config_dict)
        
        return config
    
    def _config_to_dict(self, config) -> dict:
        """Convert config object to dictionary recursively"""
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if not key.startswith('_'):
                    if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                        result[key] = self._config_to_dict(value)
                    else:
                        result[key] = value
            return result
        else:
            return config
    
    def _update_config_from_dict(self, config, config_dict):
        """Update config object from dictionary while preserving structure"""
        for key, value in config_dict.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if isinstance(value, dict) and hasattr(attr, '__dict__'):
                    # Recursively update nested objects
                    self._update_config_from_dict(attr, value)
                else:
                    # Update simple attributes
                    setattr(config, key, value)
            else:
                # Add missing attributes
                if isinstance(value, dict):
                    # Create a simple namespace object for nested dicts
                    from types import SimpleNamespace
                    setattr(config, key, SimpleNamespace(**value))
                else:
                    setattr(config, key, value)
    
    def _setup_output_directory(self):
        """Setup output directory after config is validated"""
        try:
            if hasattr(self.config, 'output') and hasattr(self.config.output, 'output_dir'):
                output_dir = self.config.output.output_dir
                experiment_name = getattr(self.config.output, 'experiment_name', 'experiment')
                self.output_dir = Path(output_dir) / experiment_name
            else:
                # Fallback to default
                self.output_dir = Path('training_output') / 'experiment'
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup tensorboard if enabled
            if (hasattr(self.config, 'training') and 
                hasattr(self.config.training, 'tensorboard') and 
                self.config.training.tensorboard):
                self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        except Exception as e:
            self.logger.warning(f"Failed to setup output directory: {e}")
            self.output_dir = Path('training_output') / 'experiment'
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup computing device using intelligent device management"""
        device = DeviceManager.get_optimal_device(self.config.device)
        
        # Get detailed device info
        device_info = DeviceManager.get_device_info(device)
        self.logger.info(f"Using device: {device}")
        self.logger.info(f"Device info: {device_info.get('name', 'Unknown')}")
        
        # Log memory info for CUDA devices
        if device.type == 'cuda' and 'memory_total_gb' in device_info:
            self.logger.info(f"GPU Memory: {device_info['memory_total_gb']:.1f}GB total, "
                           f"{device_info['memory_allocated_gb']:.1f}GB allocated")
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Create logger
        logger = logging.getLogger('UnifiedTrainer')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_registries(self):
        """Initialize registries for auto-discovery"""
        try:
            from ..registry import initialize_registries
            
            # Get registry configuration
            registry_config = getattr(self.config, 'registry', None)
            
            # Initialize with config settings
            if registry_config and getattr(registry_config, 'auto_scan', True):
                force_rescan = getattr(registry_config, 'force_rescan', False)
                additional_paths = {
                    'adapter_paths': getattr(registry_config, 'adapter_paths', []),
                    'trainer_paths': getattr(registry_config, 'trainer_paths', [])
                }
                initialize_registries(self.config, force_rescan, additional_paths)
                
                if getattr(registry_config, 'verbose', False):
                    from ..registry import list_available_components
                    list_available_components(verbose=True)
            else:
                self.logger.debug("Auto-scan disabled, skipping registry initialization")
                
        except ImportError:
            self.logger.debug("Registry system not available")
        except Exception as e:
            self.logger.warning(f"Registry initialization failed: {e}")
    
    def _should_use_external_trainer(self) -> bool:
        """Check if external trainer should be used"""
        if not hasattr(self.config, 'external_trainer'):
            return False
        
        external_config = self.config.external_trainer
        
        if external_config is None:
            return False
        
        # Handle both dictionary and object formats
        if isinstance(external_config, dict):
            enabled = external_config.get('enabled', False)
            name = external_config.get('name', None)
        else:
            enabled = getattr(external_config, 'enabled', False)
            name = getattr(external_config, 'name', None)
        
        # If enabled and has a registry name, try to resolve it
        if enabled and name:
            try:
                from ..registry import trainer_registry
                trainer_class = trainer_registry.get_trainer(name)
                if trainer_class:
                    self.logger.info(f"Found external trainer '{name}' in registry")
                    return True
                else:
                    self.logger.warning(f"External trainer '{name}' not found in registry")
            except ImportError:
                self.logger.debug("Registry not available for trainer lookup")
        
        return enabled
    
    def _setup_reproducibility(self):
        """Setup reproducibility"""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
            
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            self.logger.info(f"Reproducibility seed set to: {self.config.seed}")
    
    def setup_clean_logging(self):
        """Setup clean logging without emojis"""
        self.start_time = time.time()
        self.best_acc = 0.0
        
        # Create simple log file
        log_file = self.output_dir / "training.log"
        self.log_file = open(log_file, 'w')
        
        print(f"\n{'=' * 60}")
        print(f"TRAINING STARTED")
        print(f"{'=' * 60}")
        print(f"Model: {self.config.model.name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Output: {self.output_dir}")
        print(f"{'-' * 60}")

    def log_epoch_results(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Clean epoch logging without emojis"""
        is_best = val_acc > self.best_acc
        if is_best:
            self.best_acc = val_acc
        
        # Console output - epoch is 0-based from loop, display as 1-based
        epoch_display = epoch + 1
        status = "NEW BEST" if is_best else "        "
        print(f"Epoch {epoch_display:3d} | Train: {train_loss:.4f} ({train_acc:5.1f}%) | "
              f"Val: {val_loss:.4f} ({val_acc:5.1f}%) | {status}")
        
        # File output
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_file.write(f"{timestamp} | Epoch {epoch_display:3d} | "
                           f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
                           f"Val: {val_loss:.4f}/{val_acc:.2f}% | "
                           f"Best: {self.best_acc:.2f}%\n")
        self.log_file.flush()

    def log_training_complete(self):
        """Clean completion logging"""
        total_time = time.time() - self.start_time
        print(f"{'-' * 60}")
        print(f"TRAINING COMPLETE")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"{'=' * 60}\n")
        
        self.log_file.write(f"\n=== TRAINING COMPLETE ===\n")
        self.log_file.write(f"Best Accuracy: {self.best_acc:.2f}%\n")
        self.log_file.write(f"Total Time: {total_time:.1f}s\n")
        self.log_file.close()
    
    def __del__(self):
        """Cleanup logging resources"""
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.close()
    
    def load_model(self):
        """Load and setup model"""
        model_config = self.config.model
        
        self.logger.info(f"🔧 Loading model: {model_config.name}")
        
        # Create model based on type
        if model_config.type == "torchvision":
            self._load_torchvision_model(model_config)
        elif model_config.type == "timm":
            self._load_timm_model(model_config)
        elif model_config.type == "file":
            self._load_file_model(model_config)
        elif model_config.type == "custom":
            # Custom model loading logic
            if not model_config.path:
                raise ValueError("Custom model type requires a valid path")
            self._load_file_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_config.type}")
        
        # Create wrapper with optional adapter configuration
        adapter = None
        
        # Check for external_adapter first (takes precedence)
        if hasattr(model_config, 'external_adapter') and model_config.external_adapter:
            # Use ExternalAdapterLoader for consistency
            from ..adapters.external_adapter import ExternalAdapterLoader
            try:
                # Create a config structure for the loader
                config_for_ext = type('Config', (), {'model': model_config})
                adapter = ExternalAdapterLoader.get_adapter(config_for_ext, self.model)
                logging.info(f"Using external adapter: {model_config.external_adapter.get('class_name')}")
            except Exception as e:
                logging.error(f"Failed to load external adapter: {e}, falling back to default")
        
        # If external adapter wasn't loaded, check for standard adapter config
        if adapter is None and hasattr(model_config, 'adapter') and model_config.adapter:
            # User specified an adapter
            adapter_config = model_config.adapter_config
            adapter = ModelFactory._get_adapter(model_config.adapter, adapter_config)
            
        # If no adapter specified, ModelFactory will use AutoAdapter by default
        
        self.wrapper = ModelFactory.create_wrapped_model(
            self.model,
            adapter=adapter,
            num_classes=model_config.num_classes
        )
        
        # Move to device
        self.wrapper = self.wrapper.to(self.device)
        
        # Load pretrained weights if specified
        if model_config.pretrained_weights:
            self._load_pretrained_weights(model_config)
        
        # Freeze backbone if requested
        if model_config.freeze_backbone and hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone(freeze=True)
            self.logger.info("🧊 Backbone frozen")
        
        # Log model info
        total_params = sum(p.numel() for p in self.wrapper.parameters())
        trainable_params = sum(p.numel() for p in self.wrapper.parameters() if p.requires_grad)
        
        self.logger.info(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _load_torchvision_model(self, model_config: ModelConfig):
        """Load model from torchvision"""
        import torchvision.models as models
        
        if not hasattr(models, model_config.architecture):
            raise ValueError(f"Unknown torchvision model: {model_config.architecture}")
        
        model_class = getattr(models, model_config.architecture)
        
        # Get model args
        model_args = model_config.model_args.copy()
        if model_config.pretrained:
            model_args['pretrained'] = True
        
        self.model = model_class(**model_args)
        
        # Modify classifier for correct number of classes
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, model_config.num_classes)
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Linear):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, model_config.num_classes)
            elif isinstance(self.model.classifier, nn.Sequential):
                # Find the last Linear layer
                for i, layer in enumerate(reversed(self.model.classifier)):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        self.model.classifier[-1-i] = nn.Linear(in_features, model_config.num_classes)
                        break
        
        self.logger.info(f"✅ Loaded {model_config.architecture} from torchvision")
    
    def _load_timm_model(self, model_config: ModelConfig):
        """Load model from timm"""
        try:
            import timm
        except ImportError:
            raise ImportError("timm not installed. Install with: pip install timm")
        
        model_args = model_config.model_args.copy()
        model_args['num_classes'] = model_config.num_classes
        if model_config.pretrained:
            model_args['pretrained'] = True
        
        self.model = timm.create_model(model_config.architecture, **model_args)
        self.logger.info(f"✅ Loaded {model_config.architecture} from timm")
    
    def _load_file_model(self, model_config: ModelConfig):
        """Load model from file"""
        import importlib.util
        
        model_path = Path(model_config.path)
        self.logger.info(f"🔍 Looking for model file: {model_path}")
        
        if not model_path.exists():
            self.logger.error(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"✅ Found model file: {model_path}")
        
        try:
            # Load module
            spec = importlib.util.spec_from_file_location("custom_model", model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.logger.info(f"✅ Successfully loaded module from {model_path}")
            
            # Get model class
            if model_config.class_name:
                self.logger.info(f"🔍 Looking for class: {model_config.class_name}")
                if not hasattr(module, model_config.class_name):
                    self.logger.error(f"❌ Class {model_config.class_name} not found in {model_path}")
                    available_classes = [name for name in dir(module) if not name.startswith('_')]
                    self.logger.info(f"Available items in module: {available_classes}")
                    raise AttributeError(f"Class {model_config.class_name} not found in {model_path}")
                model_class = getattr(module, model_config.class_name)
                self.logger.info(f"✅ Found model class: {model_class}")
            else:
                # Try to auto-detect model class
                model_classes = [getattr(module, name) for name in dir(module)
                               if isinstance(getattr(module, name), type) and 
                               issubclass(getattr(module, name), nn.Module) and
                               getattr(module, name) != nn.Module]
                
                if len(model_classes) == 0:
                    raise ValueError(f"No PyTorch model classes found in {model_path}")
                elif len(model_classes) > 1:
                    class_names = [cls.__name__ for cls in model_classes]
                    raise ValueError(f"Multiple model classes found: {class_names}. Please specify class_name")
                else:
                    model_class = model_classes[0]
                    self.logger.info(f"Auto-detected model class: {model_class.__name__}")
            
            # Create model
            model_args = model_config.model_args.copy()
            if 'num_classes' not in model_args:
                model_args['num_classes'] = model_config.num_classes
            
            self.logger.info(f"🔧 Creating model with args: {model_args}")
            self.model = model_class(**model_args)
            self.logger.info(f"✅ Loaded {model_class.__name__} from {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_pretrained_weights(self, model_config: ModelConfig):
        """Load pretrained weights from checkpoint"""
        weights_path = Path(model_config.pretrained_weights)
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
        
        self.logger.info(f"🔄 Loading pretrained weights from: {weights_path}")
        
        try:
            # Load checkpoint
            if self.device.type == 'cuda':
                checkpoint = torch.load(weights_path)
            else:
                checkpoint = torch.load(weights_path, map_location='cpu')
            
            # Determine what to load based on checkpoint structure
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Checkpoint from unified training system
                    state_dict = checkpoint['model_state_dict']
                    self.logger.info("📦 Loading from unified training checkpoint")
                    
                    # Store optimizer and scheduler states for later use
                    if model_config.load_optimizer and 'optimizer_state_dict' in checkpoint:
                        self._stored_optimizer_state = checkpoint['optimizer_state_dict']
                        self.logger.info("💾 Optimizer state will be restored")
                    
                    if model_config.load_scheduler and 'scheduler_state_dict' in checkpoint:
                        self._stored_scheduler_state = checkpoint['scheduler_state_dict']
                        self.logger.info("⏰ Scheduler state will be restored")
                        
                elif 'state_dict' in checkpoint:
                    # Standard pytorch checkpoint
                    state_dict = checkpoint['state_dict']
                    self.logger.info("📦 Loading from standard pytorch checkpoint")
                else:
                    # Assume the whole dict is the state dict
                    state_dict = checkpoint
                    self.logger.info("📦 Loading direct state dict")
            else:
                # Direct state dict
                state_dict = checkpoint
                self.logger.info("📦 Loading direct state dict")
            
            # Load weights with appropriate strictness
            if model_config.strict_loading:
                missing_keys, unexpected_keys = self.wrapper.load_state_dict(state_dict, strict=True)
                self.logger.info("✅ Pretrained weights loaded successfully (strict mode)")
            else:
                missing_keys, unexpected_keys = self.wrapper.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    self.logger.warning(f"⚠️ Missing keys in pretrained weights: {len(missing_keys)} keys")
                    self.logger.debug(f"Missing keys: {missing_keys}")
                
                if unexpected_keys:
                    self.logger.warning(f"⚠️ Unexpected keys in pretrained weights: {len(unexpected_keys)} keys")
                    self.logger.debug(f"Unexpected keys: {unexpected_keys}")
                
                self.logger.info("✅ Pretrained weights loaded successfully (non-strict mode)")
                
        except Exception as e:
            error_msg = f"❌ Failed to load pretrained weights: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_transforms(self, mode: str = 'train'):
        """Create image transformations"""
        img_size = self.config.data.img_size
        aug_config = self.config.data.augmentation
        
        # Check if augmentation is enabled
        augmentation_enabled = aug_config.get('enabled', True)
        
        if mode == 'train' and augmentation_enabled:
            transforms_list = [transforms.Resize((img_size, img_size))]
            
            # Random crop (before resize if enabled)
            if aug_config.get('random_crop', False):
                crop_size = aug_config.get('random_crop_size', int(img_size * 0.9))
                transforms_list = [transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0))]
            
            # Horizontal flip
            if aug_config.get('horizontal_flip', False):
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            # Vertical flip
            if aug_config.get('vertical_flip', False):
                transforms_list.append(transforms.RandomVerticalFlip(p=0.5))
            
            # Rotation
            if aug_config.get('rotation', 0) > 0:
                transforms_list.append(transforms.RandomRotation(degrees=aug_config['rotation']))
            
            # Color jitter (legacy support with individual controls)
            if aug_config.get('color_jitter', False):
                # Use individual values if specified, otherwise use defaults
                brightness = aug_config.get('brightness', 0.2)
                contrast = aug_config.get('contrast', 0.2)
                saturation = aug_config.get('saturation', 0.2)
                hue = aug_config.get('hue', 0.0)
                transforms_list.append(transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, 
                    saturation=saturation, hue=hue
                ))
            else:
                # Individual color adjustments (when color_jitter is disabled)
                color_params = {}
                if aug_config.get('brightness', 0) > 0:
                    color_params['brightness'] = aug_config['brightness']
                if aug_config.get('contrast', 0) > 0:
                    color_params['contrast'] = aug_config['contrast']
                if aug_config.get('saturation', 0) > 0:
                    color_params['saturation'] = aug_config['saturation']
                if aug_config.get('hue', 0) > 0:
                    color_params['hue'] = aug_config['hue']
                
                if color_params:
                    transforms_list.append(transforms.ColorJitter(**color_params))
            
            # Gaussian blur
            if aug_config.get('gaussian_blur', False):
                kernel_size = aug_config.get('blur_kernel_size', 3)
                sigma = aug_config.get('blur_sigma', (0.1, 2.0))
                transforms_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
            
            transforms_list.append(transforms.ToTensor())
            
            if aug_config.get('normalize', True):
                # Use configurable normalization values with defaults
                norm_mean = aug_config.get('normalization_mean', [0.485, 0.456, 0.406])
                norm_std = aug_config.get('normalization_std', [0.229, 0.224, 0.225])
                transforms_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        
        else:  # val/test or augmentation disabled
            transforms_list = []
            
            # Center crop for validation/test if specified
            if aug_config.get('center_crop', False):
                crop_size = aug_config.get('center_crop_size', img_size)
                transforms_list.append(transforms.CenterCrop(crop_size))
            
            transforms_list.extend([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            
            if aug_config.get('normalize', True):
                # Use configurable normalization values with defaults
                norm_mean = aug_config.get('normalization_mean', [0.485, 0.456, 0.406])
                norm_std = aug_config.get('normalization_std', [0.229, 0.224, 0.225])
                transforms_list.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        
        return transforms.Compose(transforms_list)
    
    def load_data(self):
        """Load datasets and create data loaders"""
        self.logger.info("📚 Loading datasets...")
        
        # Create transforms
        train_transform = self.create_transforms('train')
        val_transform = self.create_transforms('val')
        
        # Load datasets
        train_dataset = UnifiedDataset(self.config.data, 'train', train_transform)
        val_dataset = UnifiedDataset(self.config.data, 'val', val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=self.config.data.shuffle_train,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory and self.device.type == 'cuda'
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory and self.device.type == 'cuda'
        )
        
        # Load test dataset if specified
        if self.config.data.test_path:
            test_dataset = UnifiedDataset(self.config.data, 'test', val_transform)
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory and self.device.type == 'cuda'
            )
        
        self.logger.info(f"✅ Train loader: {len(self.train_loader)} batches")
        self.logger.info(f"✅ Val loader: {len(self.val_loader)} batches")
        if self.test_loader:
            self.logger.info(f"✅ Test loader: {len(self.test_loader)} batches")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        training_config = self.config.training
        
        # Setup loss function (criterion)
        if hasattr(self.config.model, 'num_classes') and self.config.model.num_classes > 1:
            self.criterion = torch.nn.CrossEntropyLoss()
            self.logger.info(f"✅ Loss function: CrossEntropyLoss for {self.config.model.num_classes} classes")
        else:
            # Binary classification or regression
            if hasattr(self.config.data, 'class_mapping') and len(self.config.data.class_mapping) == 2:
                self.criterion = torch.nn.CrossEntropyLoss()
                self.logger.info(f"✅ Loss function: CrossEntropyLoss for binary classification")
            else:
                self.criterion = torch.nn.MSELoss()
                self.logger.info(f"✅ Loss function: MSELoss for regression")
        
        # Setup optimizer
        if training_config.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.wrapper.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.wrapper.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.wrapper.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config.optimizer}")
        
        # Setup scheduler
        if training_config.scheduler.lower() == "cosine":
            # Get T_max from scheduler_params or default to epochs
            scheduler_params = {}
            if hasattr(training_config, 'scheduler_params') and training_config.scheduler_params:
                scheduler_params = training_config.scheduler_params.copy()
                print(f"🔧 DEBUG: scheduler_params = {scheduler_params}")
            
            T_max = scheduler_params.pop('T_max', training_config.epochs)
            eta_min = scheduler_params.pop('eta_min', training_config.learning_rate * 0.01)
            
            print(f"🔧 DEBUG: T_max = {T_max}, eta_min = {eta_min}, remaining params = {scheduler_params}")
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
                **scheduler_params
            )
        elif training_config.scheduler.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.scheduler_params.get('step_size', 10),
                gamma=training_config.scheduler_params.get('gamma', 0.1)
            )
        elif training_config.scheduler.lower() == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=training_config.scheduler_params.get('factor', 0.5),
                patience=training_config.scheduler_params.get('patience', 5)
            )
        elif training_config.scheduler.lower() == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {training_config.scheduler}")
        
        self.logger.info(f"✅ Optimizer: {training_config.optimizer}")
        self.logger.info(f"✅ Scheduler: {training_config.scheduler}")
        
        # Restore optimizer state if loaded from checkpoint
        if hasattr(self, '_stored_optimizer_state') and self._stored_optimizer_state:
            try:
                self.optimizer.load_state_dict(self._stored_optimizer_state)
                self.logger.info("✅ Optimizer state restored from checkpoint")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to restore optimizer state: {e}")
        
        # Restore scheduler state if loaded from checkpoint  
        if hasattr(self, '_stored_scheduler_state') and self._stored_scheduler_state and self.scheduler:
            try:
                self.scheduler.load_state_dict(self._stored_scheduler_state)
                self.logger.info("✅ Scheduler state restored from checkpoint")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to restore scheduler state: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with performance optimizations"""
        self.wrapper.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Memory optimization at epoch start
        if self.current_epoch == 0:
            self.memory_tracker.set_baseline()
            log_memory_usage("epoch_start")
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Efficient data movement
            if self.non_blocking_transfer:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:
                images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.wrapper(images)
            loss = self.wrapper.compute_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.wrapper.parameters(),
                    max_norm=self.config.training.gradient_clipping
                )
            
            self.optimizer.step()
            
            # Essential statistics (always computed)
            train_loss += loss.item()
            
            # Detailed metrics (computed periodically for performance)
            compute_detailed_metrics = (
                self.metrics_frequency == 0 or  # Always compute if set to 0
                batch_idx % max(self.metrics_frequency, 1) == 0 or
                batch_idx == len(self.train_loader) - 1  # Always compute for last batch
            )
            
            if compute_detailed_metrics:
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                current_accuracy = 100. * train_correct / train_total if train_total > 0 else 0.0
                
                # Log progress
                if batch_idx % self.config.training.log_interval == 0:
                    self.logger.info(
                        f"  Batch {batch_idx}/{len(self.train_loader)} - "
                        f"Loss: {loss.item():.4f} - "
                        f"Acc: {current_accuracy:.2f}%"
                    )
                
                # Tensorboard logging
                if self.writer and batch_idx % self.config.training.log_interval == 0:
                    global_step = self.current_epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                    self.writer.add_scalar('Accuracy/Train_Batch', current_accuracy, global_step)
            else:
                # Lightweight logging for non-detailed steps
                if batch_idx % (self.config.training.log_interval * 2) == 0:
                    self.logger.debug(f"  Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
        
        # Final metrics calculation if not computed in last batch
        if not compute_detailed_metrics:
            with torch.no_grad():
                # Quick final pass for accuracy if needed
                if train_total == 0:
                    for images, labels in self.train_loader:
                        images = images.to(self.device, non_blocking=self.non_blocking_transfer)
                        labels = labels.to(self.device, non_blocking=self.non_blocking_transfer)
                        outputs = self.wrapper(images)
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                        if train_total >= 1000:  # Sample for efficiency
                            break
        
        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100. * train_correct / train_total if train_total > 0 else 0.0
        
        # Memory tracking
        self.memory_tracker.check_peak()
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate model with memory efficiency"""
        self.wrapper.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        if self.efficient_validation:
            # Memory-efficient validation
            with torch.no_grad():
                for images, labels in self.val_loader:
                    # Efficient data movement
                    if self.non_blocking_transfer:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    else:
                        images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.wrapper(images)
                    loss = self.wrapper.compute_loss(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Clear intermediate variables for memory efficiency
                    del outputs, predicted
        else:
            # Original validation (backward compatibility)
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.wrapper(images)
                    loss = self.wrapper.compute_loss(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
        
        # Back to training mode
        self.wrapper.train()
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * val_correct / val_total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_weight_extension(self) -> str:
        """Determine the appropriate weight file extension"""
        if self.config.output.weight_format == "auto":
            # Default to pth for auto mode
            return 'pth'
        elif self.config.output.weight_format == "pt":
            return 'pt'
        elif self.config.output.weight_format == "pth":
            return 'pth'
        else:
            # Default to pth for unknown formats
            self.logger.warning(f"Unknown weight format: {self.config.output.weight_format}, using .pth")
            return 'pth'

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint with performance optimizations"""
        # Get the appropriate extension
        ext = self.get_weight_extension()
        
        # Save lightweight model weights for deployment (if enabled)
        if self.config.output.save_lightweight and is_best:
            lightweight_path = self.output_dir / f'model_weights.{ext}'
            torch.save(self.wrapper.state_dict(), lightweight_path)
            self.logger.info(f"💾 Saved lightweight model: {lightweight_path.name}")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.wrapper.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config.to_dict()
        }
        
        # Add optimizer state if configured
        if self.config.output.save_optimizer and self.optimizer:
            checkpoint_data['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Add scheduler state if configured
        if self.config.output.save_scheduler and self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add training history (keep recent only for performance)
        keep_recent = self.config.output.keep_recent_checkpoints
        if keep_recent > 0:
            checkpoint_data['train_losses'] = self.train_losses[-100:] if len(self.train_losses) > 100 else self.train_losses
            checkpoint_data['val_accuracies'] = self.val_accuracies[-20:] if len(self.val_accuracies) > 20 else self.val_accuracies
        else:
            checkpoint_data['train_losses'] = self.train_losses
            checkpoint_data['val_accuracies'] = self.val_accuracies
        
        # Save latest checkpoint
        if self.config.output.save_last:
            torch.save(checkpoint_data, self.output_dir / f'last_checkpoint.{ext}')
        
        # Save best checkpoint
        if is_best and self.config.output.save_best_only:
            torch.save(checkpoint_data, self.output_dir / f'best_checkpoint.{ext}')
            self.logger.info(f"💾 Saved best checkpoint: best_checkpoint.{ext}")
        
        # Save periodic full checkpoint (for training resumption)
        checkpoint_freq = self.config.training.checkpoint_frequency
        if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
            periodic_path = self.output_dir / f'checkpoint_epoch_{epoch}.{ext}'
            torch.save(checkpoint_data, periodic_path)
            self.logger.info(f"💾 Saved periodic checkpoint: {periodic_path.name}")
        
        # Clean up old checkpoints if configured
        self._cleanup_old_checkpoints(keep_recent)
    
    def _cleanup_old_checkpoints(self, keep_recent: int):
        """Clean up old checkpoint files to save disk space"""
        if keep_recent <= 0:
            return
        
        try:
            # Find all checkpoint files
            checkpoint_pattern = "checkpoint_epoch_*.pt*"
            checkpoint_files = list(self.output_dir.glob(checkpoint_pattern))
            
            if len(checkpoint_files) > keep_recent:
                # Sort by epoch number
                checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                
                # Remove oldest files
                files_to_remove = checkpoint_files[:-keep_recent]
                for file_path in files_to_remove:
                    file_path.unlink()
                    self.logger.debug(f"🗑️  Removed old checkpoint: {file_path.name}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to cleanup old checkpoints: {e}")
    
    def _resume_from_checkpoint(self, checkpoint_path: Path) -> int:
        """Resume training from a checkpoint file"""
        try:
            self.logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.wrapper.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("✅ Model state loaded successfully")
            else:
                self.logger.warning("⚠️  No model state found in checkpoint")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✅ Optimizer state loaded successfully")
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("✅ Scheduler state loaded successfully")
            
            # Load best validation accuracy
            if 'best_val_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_val_acc']
                self.logger.info(f"✅ Best validation accuracy: {self.best_val_acc:.4f}")
            
            # Load training history
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_accuracies' in checkpoint:
                self.val_accuracies = checkpoint['val_accuracies']
            
            # Get starting epoch
            start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
            self.logger.info(f"✅ Resuming training from epoch {start_epoch}")
            
            return start_epoch
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            self.logger.warning("🔄 Starting training from scratch")
            return 0
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file"""
        checkpoint_patterns = [
            "last_checkpoint.*",
            "checkpoint_epoch_*.*", 
            "best_checkpoint.*"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in checkpoint_patterns:
            for checkpoint_file in self.output_dir.glob(pattern):
                if checkpoint_file.stat().st_mtime > latest_time:
                    latest_time = checkpoint_file.stat().st_mtime
                    latest_file = checkpoint_file
        
        return latest_file
    
    def _load_checkpoint_and_resume(self, checkpoint_path: Path) -> int:
        """Load checkpoint and return start epoch"""
        try:
            self.logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.wrapper.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("✅ Model state loaded successfully")
            
            # Load optimizer state (after optimizer is created)
            if 'optimizer_state_dict' in checkpoint and self.optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✅ Optimizer state loaded successfully")
            
            # Load scheduler state (after scheduler is created)
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("✅ Scheduler state loaded successfully")
            
            # Load best validation accuracy
            if 'best_val_acc' in checkpoint:
                self.best_val_acc = checkpoint['best_val_acc']
                self.logger.info(f"✅ Best validation accuracy: {self.best_val_acc:.4f}")
            
            # Load training history
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_accuracies' in checkpoint:
                self.val_accuracies = checkpoint['val_accuracies']
            
            # CORRECTED: Get starting epoch
            # Checkpoint stores the completed epoch (1-based)
            # We need to return the next epoch to train (0-based for the loop)
            completed_epoch = checkpoint.get('epoch', 0)  # This is 1-based (human readable)
            start_epoch = completed_epoch  # Use completed epoch as 0-based loop index
            
            self.logger.info(f"✅ Checkpoint was saved after completing epoch {completed_epoch}")
            self.logger.info(f"✅ Resuming training from epoch {start_epoch + 1} (will be displayed as epoch {start_epoch + 1})")
            
            return start_epoch
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            self.logger.warning("🔄 Starting training from scratch")
            return 0
    
    def train(self):
        """Main training loop - handles both built-in and external trainers"""
        
        # Check if external trainer should be used
        if self.use_external_trainer:
            return self._train_with_external_trainer()
        else:
            return self._train_with_builtin_trainer()
    
    def _train_with_external_trainer(self):
        """Train using external trainer"""
        self.logger.info("🔧 Using External Training Method")
        self.logger.info("=" * 60)
        
        # For external trainers, skip model and data loading
        # The external trainer will handle these
        
        # Create HybridTrainer which handles external training
        hybrid_trainer = HybridTrainer(
            config=self.config,
            model=None,  # External trainer will create its own model
            device=self.device,
            output_dir=self.output_dir
        )
        
        # Delegate training to external trainer
        return hybrid_trainer.train()
    
    def _train_with_builtin_trainer(self):
        """Train using built-in training method with clean logging"""
        # Setup everything
        self.load_model()
        self.load_data()
        self.setup_optimizer()
        
        # Check for resume from checkpoint
        start_epoch = 0
        if hasattr(self.config.training, 'resume_from_checkpoint') and self.config.training.resume_from_checkpoint:
            resume_config = self.config.training.resume_from_checkpoint
            
            if resume_config is True:
                # Auto-find latest checkpoint
                checkpoint_path = self._find_latest_checkpoint()
                if checkpoint_path:
                    start_epoch = self._load_checkpoint_and_resume(checkpoint_path)
                else:
                    self.logger.warning("🔍 No checkpoint found for auto-resume, starting from scratch")
            elif isinstance(resume_config, str):
                # Use specified checkpoint path
                checkpoint_path = Path(resume_config)
                if checkpoint_path.exists():
                    start_epoch = self._load_checkpoint_and_resume(checkpoint_path)
                else:
                    self.logger.error(f"❌ Specified checkpoint not found: {checkpoint_path}")
                    self.logger.warning("🔄 Starting training from scratch")
        
        self.setup_clean_logging()
        
        # Save configuration
        if self.config.output.save_config:
            self.config.save_yaml(self.output_dir / 'config.yaml')
            self.config.save_json(self.output_dir / 'config.json')
        
        # Show dataset info after loading
        print(f"Dataset: {len(self.train_loader):,} train batches, {len(self.val_loader):,} val batches")
        
        epochs = self.config.training.epochs
        
        try:
            for epoch in range(start_epoch, epochs):
                self.current_epoch = epoch
                
                # Training phase with progress bar
                train_loss, train_acc = self._train_epoch_with_progress(epoch)
                self.train_losses.append(train_loss)
                
                # Validation phase with progress bar
                val_loss, val_acc = self._validate_epoch_with_progress(epoch)
                self.val_accuracies.append(val_acc)
                
                # Scheduler step
                if self.scheduler:
                    if self.config.training.scheduler.lower() == 'plateau':
                        self.scheduler.step(val_acc)
                    else:
                        self.scheduler.step()
                
                # Log results
                self.log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc)
                
                # Update best accuracy
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                
                # Save checkpoint with correct epoch numbering
                # epoch is 0-based in loop, save as epoch + 1 (completed epoch, 1-based)
                if epoch % self.config.training.save_frequency == 0 or val_acc > self.best_val_acc:
                    self.save_checkpoint(epoch + 1, val_acc > self.best_val_acc)
                
                # Tensorboard logging
                if self.writer:
                    self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
                    self.writer.add_scalar('Loss/Val_Epoch', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/Train_Epoch', train_acc, epoch)
                    self.writer.add_scalar('Accuracy/Val_Epoch', val_acc, epoch)
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            self.log_training_complete()
            
            # Save final results
            results = {
                'best_val_accuracy': self.best_val_acc,
                'train_losses': self.train_losses,
                'val_accuracies': self.val_accuracies,
                'config': self.config.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Close tensorboard writer
            if self.writer:
                self.writer.close()
            
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
                self.log_file.close()
            raise
    
    def _train_epoch_with_progress(self, epoch):
        """Training epoch with progress bar"""
        self.wrapper.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        # epoch is 0-based from the loop, but we display 1-based for humans
        epoch_display = epoch + 1
        desc = f"Epoch {epoch_display:2d} Train"
        pbar = tqdm(self.train_loader, desc=desc, leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.wrapper(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.1f}%'
            })
        
        pbar.close()
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _validate_epoch_with_progress(self, epoch):
        """Validation epoch with progress bar"""
        self.wrapper.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for validation
        # epoch is 0-based from the loop, but we display 1-based for humans
        epoch_display = epoch + 1
        desc = f"Epoch {epoch_display:2d} Val  "
        pbar = tqdm(self.val_loader, desc=desc, leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.wrapper(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Update progress bar
                current_acc = 100.0 * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
        
        pbar.close()
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
