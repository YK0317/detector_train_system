"""
Configuration management for Train System

This module defines the configuration schema and management for the training system.
All models can be trained using this unified configuration format.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import json
import logging


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "model"
    type: str = "file"  # "file", "torchvision", "timm", "custom"
    path: Optional[str] = None  # Path to model file
    class_name: Optional[str] = None  # Class name in the file
    architecture: Optional[str] = None  # For torchvision/timm models
    pretrained: bool = True
    num_classes: int = 2
    
    # Pretrained weights configuration
    pretrained_weights: Optional[str] = None  # Path to pretrained weights (.pth, .pt file)
    strict_loading: bool = True  # Whether to strictly load all parameters
    load_optimizer: bool = False  # Whether to load optimizer state from checkpoint
    load_scheduler: bool = False  # Whether to load scheduler state from checkpoint
    
    # Model-specific parameters
    img_size: int = 224
    dropout: float = 0.1
    freeze_backbone: bool = False
    
    # Additional model arguments
    model_args: Dict[str, Any] = field(default_factory=dict)
    
    # Adapter configuration (optional, defaults to AutoAdapter)
    adapter: Optional[str] = None  # "auto", "standard", "logits_features", "dict", "external", etc.
    adapter_config: Dict[str, Any] = field(default_factory=dict)  # Additional adapter parameters
    # For external adapter: adapter="external", adapter_config={"script_path": "path/to/adapter.py", "class_name": "MyAdapter"}
    
    # External adapter configuration (alternative to adapter="external")
    external_adapter: Optional[Dict[str, Any]] = None  # {"script_path": "path/to/adapter.py", "class_name": "MyAdapter"}


@dataclass
class DataConfig:
    """Dataset configuration"""
    name: str = "dataset"
    type: str = "folder"  # "folder", "class_folders", "csv", "json", "custom"
    train_path: str = ""
    val_path: str = ""
    test_path: Optional[str] = None
    
    # Class folder configuration (for type="class_folders")
    class_folders: Dict[str, str] = field(default_factory=dict)  # {"class_name": "path"}
    real_path: Optional[str] = None  # Direct path to real class folder
    fake_path: Optional[str] = None  # Direct path to fake class folder
    
    # Data processing
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    # Data augmentation
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,  # Master switch to enable/disable all augmentations
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 10,
        "color_jitter": True,  # Legacy combined color adjustments
        "brightness": 0.0,     # Individual color controls (when color_jitter=False)
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
        "gaussian_blur": False,
        "blur_kernel_size": 3,
        "blur_sigma": [0.1, 2.0],
        "random_crop": False,
        "random_crop_size": None,
        "center_crop": False,
        "center_crop_size": None,
        "normalize": True,
        "normalization_mean": [0.485, 0.456, 0.406],  # ImageNet defaults
        "normalization_std": [0.229, 0.224, 0.225]    # ImageNet defaults
    })
    
    # Dataset-specific parameters
    max_samples: Optional[int] = None
    class_mapping: Dict[str, int] = field(default_factory=lambda: {"real": 0, "fake": 1})


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    
    # Training settings
    gradient_clipping: float = 1.0
    mixed_precision: bool = False
    accumulation_steps: int = 1
    
    # Validation and saving
    val_frequency: int = 1
    save_frequency: int = 5
    early_stopping_patience: int = 10
    
    # Performance optimizations
    metrics_frequency: int = 100  # Calculate detailed metrics every N steps (0 = every step)
    checkpoint_frequency: int = 5  # Save full checkpoint every N epochs
    non_blocking_transfer: bool = True  # Use non_blocking data transfer
    efficient_validation: bool = True  # Use memory-efficient validation
    
    # Logging
    log_interval: int = 10
    tensorboard: bool = True
    wandb: bool = False
    
    # Scheduler parameters
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "training_output"
    experiment_name: str = "experiment"
    save_best_only: bool = True
    save_last: bool = True
    
    # What to save
    save_model: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_logs: bool = True
    save_config: bool = True
    
    # Weight file format
    weight_format: str = "auto"  # "auto", "pth", "pt"
    
    # Performance optimizations
    save_lightweight: bool = True  # Save lightweight model weights for deployment
    keep_recent_checkpoints: int = 3  # Only keep N recent full checkpoints (0 = keep all)


@dataclass
class ExternalTrainerConfig:
    """Configuration for external training methods"""
    enabled: bool = False
    script_path: Optional[str] = None  # Path to external training script
    class_name: Optional[str] = None   # Name of trainer class (auto-detect if None)
    name: Optional[str] = None         # Registry name for auto-discovery (NEW)
    
    # Parameters to pass to external trainer
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Override settings
    override_optimizer: bool = False    # Whether external trainer handles optimizer
    override_scheduler: bool = False    # Whether external trainer handles scheduler
    override_loss: bool = False         # Whether external trainer handles loss function
    override_saving: bool = False       # Whether external trainer handles checkpoint saving


@dataclass 
class RegistryConfig:
    """Registry configuration for auto-discovery"""
    adapter_paths: List[str] = field(default_factory=list)
    trainer_paths: List[str] = field(default_factory=list)
    auto_scan: bool = True
    verbose: bool = False
    force_rescan: bool = False


@dataclass
class UnifiedTrainingConfig:
    """Complete unified training configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # External trainer configuration
    external_trainer: Optional[ExternalTrainerConfig] = field(default_factory=lambda: ExternalTrainerConfig())
    
    # Registry configuration for auto-discovery (NEW)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    
    # System settings
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0"
    seed: int = 42
    deterministic: bool = False
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'UnifiedTrainingConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'UnifiedTrainingConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedTrainingConfig':
        """Create configuration from dictionary"""
        # Extract nested configurations
        model_data = data.get('model', {})
        data_config_data = data.get('data', {})
        training_data = data.get('training', {})
        output_data = data.get('output', {})
        external_trainer_data = data.get('external_trainer', {})
        
        # Create nested config objects
        model_config = ModelConfig(**model_data)
        data_config = DataConfig(**data_config_data)
        training_config = TrainingConfig(**training_data)
        output_config = OutputConfig(**output_data)
        external_trainer_config = ExternalTrainerConfig(**external_trainer_data)
        
        # Create main config
        main_data = {k: v for k, v in data.items() 
                    if k not in ['model', 'data', 'training', 'output', 'external_trainer']}
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            output=output_config,
            external_trainer=external_trainer_config,
            **main_data
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'output': self.output.__dict__,
            'external_trainer': self.external_trainer.__dict__ if self.external_trainer else {},
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'description': self.description,
            'tags': self.tags
        }
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def save_json(self, json_path: Union[str, Path]):
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigValidator:
    """Configuration validator"""
    
    @staticmethod
    def validate(config: UnifiedTrainingConfig) -> 'ValidationResult':
        """Validate configuration"""
        errors = []
        warnings = []
        
        # Model validation
        if config.model.type == "file":
            if not config.model.path:
                errors.append("Model path is required when type is 'file'")
            elif not Path(config.model.path).exists():
                errors.append(f"Model file not found: {config.model.path}")
            
            if not config.model.class_name:
                warnings.append("Model class_name not specified, will try to auto-detect")
        
        # Data validation
        if config.data.type == "folder":
            if not config.data.train_path:
                errors.append("Train path is required")
            elif not Path(config.data.train_path).exists():
                errors.append(f"Train path not found: {config.data.train_path}")
            
            if config.data.val_path and not Path(config.data.val_path).exists():
                warnings.append(f"Validation path not found: {config.data.val_path}")
        
        # Training validation
        if config.training.epochs <= 0:
            errors.append("Epochs must be positive")
        
        if config.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        # Output validation
        output_dir = Path(config.output.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
        
        return ValidationResult(errors=errors, warnings=warnings)


@dataclass
class ValidationResult:
    """Configuration validation result"""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.errors) == 0
    
    def log_results(self, logger: Optional[logging.Logger] = None):
        """Log validation results"""
        if logger is None:
            logger = logging.getLogger(__name__)
        
        if self.errors:
            for error in self.errors:
                logger.error(f"❌ {error}")
        
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"⚠️ {warning}")
        
        if self.is_valid:
            logger.info("✅ Configuration is valid")


class ConfigTemplateManager:
    """Manager for configuration templates"""
    
    # Template definitions
    BLIP_CONFIG_TEMPLATE = {
        "model": {
            "name": "BLIP",
            "type": "file",
            "path": "../models/standalone_blip_deepfake.py",
            "class_name": "BLIPDeepfakeDetector",
            "img_size": 384,
            "dropout": 0.1,
            "freeze_backbone": True,
            "pretrained_weights": None,
            "strict_loading": True,
            "load_optimizer": False,
            "load_scheduler": False,
            "model_args": {
                "vit_arch": "base",
                "drop_path_rate": 0.1
            }
        },
        "data": {
            "name": "URS",
            "type": "folder",
            "train_path": "URS-train/train",
            "val_path": "URS-train/validation",
            "img_size": 384,
            "batch_size": 8,
            "num_workers": 4
        },
        "training": {
            "epochs": 20,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": "cosine"
        },
        "output": {
            "output_dir": "blip_training_output",
            "experiment_name": "blip_urs_experiment"
        }
    }
    
    GENERIC_CONFIG_TEMPLATE = {
        "model": {
            "name": "GenericModel",
            "type": "file",
            "path": "models/your_model.py",
            "class_name": "YourModelClass",
            "num_classes": 2,
            "pretrained_weights": None,
            "strict_loading": True,
            "load_optimizer": False,
            "load_scheduler": False
        },
        "data": {
            "name": "YourDataset",
            "type": "folder",
            "train_path": "data/train",
            "val_path": "data/val",
            "batch_size": 32
        },
        "training": {
            "epochs": 10,
            "learning_rate": 1e-4
        },
        "output": {
            "output_dir": "training_output",
            "experiment_name": "my_experiment"
        }
    }
    
    TORCHVISION_TEMPLATE = {
        "model": {
            "name": "ResNet18",
            "type": "torchvision",
            "architecture": "resnet18",
            "pretrained": True,
            "num_classes": 2
        },
        "data": {
            "name": "ImageDataset",
            "type": "folder",
            "train_path": "data/train",
            "val_path": "data/val",
            "batch_size": 32,
            "img_size": 224
        },
        "training": {
            "epochs": 20,
            "learning_rate": 1e-3,
            "optimizer": "sgd",
            "scheduler": "step"
        },
        "output": {
            "output_dir": "resnet_output",
            "experiment_name": "resnet18_classification"
        }
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> Dict[str, Any]:
        """Get configuration template by name"""
        templates = {
            "blip": cls.BLIP_CONFIG_TEMPLATE,
            "generic": cls.GENERIC_CONFIG_TEMPLATE,
            "torchvision": cls.TORCHVISION_TEMPLATE,
            "complete": cls._get_complete_template
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
        
        if callable(templates[template_name]):
            return templates[template_name]()
        return templates[template_name].copy()
    
    @classmethod
    def _get_complete_template(cls) -> Dict[str, Any]:
        """Get the complete configuration template with all fields"""
        from pathlib import Path
        import yaml
        
        # Path to the complete config template file
        template_path = Path(__file__).parent.parent / "configs" / "complete_config_template.yaml"
        
        if template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    content = f.read()
                    # Remove comments and example sections for clean template
                    lines = content.split('\n')
                    cleaned_lines = []
                    skip_section = False
                    
                    for line in lines:
                        # Skip comment blocks and examples
                        if line.strip().startswith('# ============================================================================'):
                            if 'EXAMPLES' in line or 'FIELD DESCRIPTIONS' in line:
                                skip_section = True
                                continue
                            else:
                                skip_section = False
                        
                        if skip_section:
                            continue
                            
                        # Remove inline comments but keep structure
                        if '#' in line and not line.strip().startswith('#'):
                            line = line.split('#')[0].rstrip()
                        
                        # Skip full comment lines but keep important structure
                        if line.strip().startswith('#'):
                            continue
                            
                        cleaned_lines.append(line)
                    
                    cleaned_content = '\n'.join(cleaned_lines)
                    template = yaml.safe_load(cleaned_content)
                    return template
            except Exception as e:
                print(f"Warning: Could not load complete template from file: {e}")
                pass
        
        # Fallback to basic complete template
        return {
            "model": {
                "name": "model_name",
                "type": "file",
                "path": "models/your_model.py",
                "class_name": "YourModelClass",
                "architecture": None,
                "pretrained": True,
                "num_classes": 2,
                "pretrained_weights": None,
                "strict_loading": True,
                "load_optimizer": False,
                "load_scheduler": False,
                "img_size": 224,
                "dropout": 0.1,
                "freeze_backbone": False,
                "model_args": {},
                "adapter": None,
                "adapter_config": {},
                "external_adapter": {
                    "script_path": None,
                    "class_name": None,
                    "required_packages": [],
                    "parameters": {}
                }
            },
            "data": {
                "name": "dataset_name",
                "type": "folder",
                "train_path": "data/train",
                "val_path": "data/val",
                "test_path": None,
                "class_folders": {},
                "real_path": None,
                "fake_path": None,
                "img_size": 224,
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": True,
                "shuffle_train": True,
                "augmentation": {
                    "enabled": True,
                    "horizontal_flip": True,
                    "rotation": 10,
                    "color_jitter": True,
                    "normalize": True,
                    "normalization_mean": [0.485, 0.456, 0.406],
                    "normalization_std": [0.229, 0.224, 0.225],
                    "vertical_flip": False,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "saturation": 0.0,
                    "hue": 0.0,
                    "gaussian_blur": False,
                    "blur_kernel_size": 3,
                    "blur_sigma": [0.1, 2.0],
                    "random_crop": False,
                    "random_crop_size": None,
                    "center_crop": False,
                    "center_crop_size": None
                },
                "max_samples": None,
                "class_mapping": {
                    "real": 0,
                    "fake": 1
                }
            },
            "training": {
                "epochs": 20,
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "gradient_clipping": 1.0,
                "mixed_precision": False,
                "accumulation_steps": 1,
                "val_frequency": 1,
                "save_frequency": 5,
                "early_stopping_patience": 10,
                "metrics_frequency": 100,
                "checkpoint_frequency": 5,
                "non_blocking_transfer": True,
                "efficient_validation": True,
                "log_interval": 10,
                "tensorboard": True,
                "wandb": False,
                "scheduler_params": {}
            },
            "output": {
                "output_dir": "training_output",
                "experiment_name": "experiment",
                "save_best_only": True,
                "save_last": True,
                "save_model": True,
                "save_optimizer": True,
                "save_scheduler": True,
                "save_logs": True,
                "save_config": True,
                "weight_format": "auto",
                "save_lightweight": True,
                "keep_recent_checkpoints": 3
            },
            "external_trainer": {
                "enabled": False,
                "script_path": None,
                "class_name": None,
                "parameters": {},
                "override_optimizer": False,
                "override_scheduler": False,
                "override_loss": False,
                "override_saving": False
            },
            "device": "auto",
            "seed": 42,
            "deterministic": False,
            "description": "Complete configuration template with all supported fields",
            "tags": ["template", "complete", "example"]
        }
    
    @classmethod
    def create_config_from_template(cls, template_name: str, **overrides) -> UnifiedTrainingConfig:
        """Create configuration from template with optional overrides"""
        template = cls.get_template(template_name)
        
        # Apply overrides
        def deep_update(d: dict, u: dict):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        if overrides:
            template = deep_update(template, overrides)
        
        return UnifiedTrainingConfig.from_dict(template)
    
    @classmethod
    def save_template(cls, template_name: str, output_path: Union[str, Path], format: str = "yaml"):
        """Save template to file"""
        template = cls.get_template(template_name)
        output_path = Path(output_path)
        
        if format.lower() == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")


# Backwards compatibility
REQUIRED_FIELDS = ["model", "data", "training", "output"]
OPTIONAL_FIELDS = ["device", "seed", "deterministic", "description", "tags"]
DEFAULT_VALUES = {
    "device": "auto",
    "seed": 42,
    "deterministic": False,
    "description": "",
    "tags": []
}
