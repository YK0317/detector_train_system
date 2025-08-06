# Train System

A comprehensive training system for PyTorch models that provides a unified interface for training any model with any dataset through configuration files and API access.

## Features

- **Universal Model Support**: Train any PyTorch model with a unified interface
- **High-Performance Training**: Optimized for speed and memory efficiency (see [Performance Optimizations](PERFORMANCE_OPTIMIZATIONS.md))
- **Flexible Adapter System**: Handles different model output formats automatically
- **Configuration-Based Training**: Define everything through YAML/JSON configuration files
- **REST API**: Remote training control via HTTP endpoints
- **Command-Line Interface**: Easy-to-use CLI for common tasks
- **Pretrained Weights Support**: Load and fine-tune from checkpoints
- **Multiple Data Formats**: Support for folder structures, CSV, and JSON datasets
- **Comprehensive Logging**: TensorBoard integration and detailed logging
- **Model Factory**: Easy model creation from torchvision, timm, or custom files
- **Memory Management**: Advanced memory tracking and optimization utilities

## Installation

### From Source

```bash
git clone <repository-url>
cd train_system
pip install -e .
```

### From PyPI (when published)

```bash
pip install train-system
```

## Quick Start

### 1. Create a Configuration Template

```bash
train-system create-template torchvision my_config.yaml
```

### 2. Edit Configuration

```yaml
model:
  name: "ResNet18"
  type: "torchvision"
  architecture: "resnet18"
  pretrained: true
  num_classes: 2

data:
  name: "MyDataset"
  type: "folder"
  train_path: "data/train"
  val_path: "data/val"
  batch_size: 32

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: "adamw"

output:
  output_dir: "experiments"
  experiment_name: "resnet18_experiment"
```

### 3. Train the Model

```bash
train-system train my_config.yaml
```

### 4. Monitor Training

```bash
# List experiments
train-system list

# View specific results
train-system results resnet18_experiment

# Start API server for remote monitoring
train-system api
```

## Performance Optimizations 🚀

The train_system includes advanced performance optimizations that significantly reduce training overhead while maintaining full backward compatibility. See [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) for complete details.

### Key Performance Features:

- **Periodic Metrics**: Calculate detailed metrics every N steps instead of every step (7.9%+ overhead reduction)
- **Memory-Efficient Validation**: Optimized validation loop with `torch.no_grad()` and smart memory management
- **Non-blocking GPU Transfer**: Faster data pipeline with non-blocking data transfer
- **Lightweight Checkpoints**: Separate deployment weights + strategic full checkpoint saving
- **CUDA Optimizations**: Automatic TF32, cuDNN benchmarking, and memory optimizations
- **Memory Tracking**: Advanced memory monitoring and automatic cleanup utilities

### Performance Configuration Example:

```yaml
training:
  metrics_frequency: 200        # Calculate metrics every 200 steps (vs every step)
  checkpoint_frequency: 10      # Full checkpoint every 10 epochs
  non_blocking_transfer: true   # Faster GPU data transfer
  efficient_validation: true   # Memory-efficient validation

output:
  save_lightweight: true       # Fast deployment weights
  keep_recent_checkpoints: 3   # Automatic cleanup of old checkpoints
```

All optimizations are **enabled by default with sensible settings** and maintain **100% backward compatibility** with existing configurations.

## Usage Examples

### Using the Python API

```python
from train_system import UnifiedTrainingWrapper, ModelFactory
from train_system.config import UnifiedTrainingConfig
from train_system.core import UnifiedTrainer
import torchvision.models as models

# Method 1: Direct model wrapping
model = models.resnet18(pretrained=True)
wrapper = ModelFactory.create_wrapped_model(model, num_classes=2)

# Method 2: Configuration-based training
config = UnifiedTrainingConfig.from_yaml("config.yaml")
trainer = UnifiedTrainer(config)
results = trainer.train()

# Method 3: Using model factory utilities
from train_system.core.wrapper import ModelUtils
wrapper = ModelUtils.create_from_torchvision("resnet18", num_classes=2)
```

### Working with Custom Models

```python
# For custom model files
wrapper = ModelUtils.create_from_custom_path(
    "models/my_model.py", 
    "MyModelClass", 
    num_classes=2
)

# Using timm models
wrapper = ModelUtils.create_from_timm("efficientnet_b4", num_classes=2)
```

### REST API Usage

```python
import requests

# Start training via API
config = {...}  # Your configuration
response = requests.post("http://localhost:5000/train", json=config)

# Check status
status = requests.get("http://localhost:5000/status").json()

# Get results
results = requests.get("http://localhost:5000/results").json()
```

## Configuration Reference

### Model Configuration

```yaml
model:
  name: "ModelName"                    # Display name
  type: "torchvision"                  # "torchvision", "timm", "file", "custom"
  architecture: "resnet18"             # For torchvision/timm
  path: "models/my_model.py"          # For custom files
  class_name: "MyModel"               # Class name in file
  pretrained: true                    # Use pretrained weights
  num_classes: 2                      # Number of output classes
  pretrained_weights: "weights.pth"   # Custom pretrained weights
  strict_loading: true                # Strict weight loading
  freeze_backbone: false              # Freeze feature extractor
  model_args:                         # Additional model arguments
    dropout: 0.1
```

### Data Configuration

```yaml
data:
  name: "MyDataset"
  type: "folder"                      # "folder", "csv", "json"
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"              # Optional
  img_size: 224
  batch_size: 32
  num_workers: 4
  shuffle_train: true
  augmentation:
    horizontal_flip: true
    rotation: 10
    color_jitter: true
    normalize: true
  class_mapping:
    real: 0
    fake: 1
```

### Training Configuration

```yaml
training:
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.01
  optimizer: "adamw"                  # "adam", "adamw", "sgd"
  scheduler: "cosine"                 # "cosine", "step", "plateau", "none"
  gradient_clipping: 1.0
  mixed_precision: false
  val_frequency: 1
  save_frequency: 5
  early_stopping_patience: 10
  tensorboard: true
  scheduler_params:
    step_size: 10
    gamma: 0.1
```

## Command Line Interface

### Available Commands

```bash
# Create configuration templates
train-system create-template <type> <output_path>

# Validate configuration
train-system validate config.yaml

# Train model
train-system train config.yaml

# List experiments
train-system list

# Show experiment results
train-system results experiment_name

# Start API server
train-system api --host 0.0.0.0 --port 5000
```

### Quick Commands

```bash
# Quick training (shorthand)
ts-train config.yaml

# Validate only
ts-train config.yaml --validate-only

# Start API server
ts-api --port 8000
```

## Supported Model Types

### Torchvision Models
- All models from `torchvision.models`
- Automatic classifier head adaptation
- Pretrained weights support

### Timm Models
- 1000+ models from the timm library
- State-of-the-art architectures
- Easy integration

### Custom Models
- Any PyTorch `nn.Module`
- Automatic output format detection
- Flexible adapter system

## Adapter System

The system automatically handles different model output formats:

- **StandardAdapter**: Single tensor output (logits)
- **LogitsAndFeaturesAdapter**: Tuple of (logits, features)
- **DictOutputAdapter**: Dictionary output
- **CapsuleNetworkAdapter**: CapsuleNet-specific outputs
- **AutoAdapter**: Automatic detection (default)

## API Endpoints

- `GET /` - API documentation
- `GET /status` - Training status
- `POST /train` - Start training
- `POST /stop` - Stop training
- `GET /results` - Get results
- `GET /templates` - List templates
- `POST /validate` - Validate config

## Output Structure

```
experiments/
├── experiment_name/
│   ├── config.yaml
│   ├── config.json
│   ├── best_checkpoint.pth
│   ├── last_checkpoint.pth
│   ├── training_results.json
│   ├── training.log
│   └── tensorboard/
│       └── events.out.tfevents...
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Additional dependencies in `requirements.txt`

## 🧪 Testing

The Train System includes a comprehensive test suite with **22 test cases** covering all major components and functionality.

### 🚀 Quick Test Run

```bash
# Run all tests
cd train_system
python run_tests.py

# Run with pytest (if installed)
python -m pytest tests/test_comprehensive.py -v

# Run tests directly
python tests/test_comprehensive.py
```

### 📊 Test Coverage Overview

| Component | Coverage | Tests |
|-----------|----------|-------|
| 🤖 Model Wrapping | 100% | 3 tests |
| ⚙️ Configuration | 100% | 4 tests |
| 📊 Data Handling | 85% | 2 tests |
| 🚀 Training Pipeline | 90% | 3 tests |
| 🛠️ Utilities | 80% | 3 tests |
| 🚨 Error Handling | 95% | 2 tests |
| 🔄 Integration | 100% | 2 tests |
| 🔧 System Imports | 100% | 3 tests |

**Total: 22 tests covering all major functionality**

### 🎯 Test Categories

#### 1. **Core System Tests**
- Import validation for all components
- Version information accessibility
- Optional dependency handling

#### 2. **Model Integration Tests**
- TorchVision model wrapping (ResNet18, MobileNetV2)
- Custom PyTorch model integration
- Adapter functionality (Tensor, Tuple, Dict outputs)
- Prediction pipeline validation

#### 3. **Configuration Tests**
- YAML/JSON config creation and validation
- Template system functionality
- File save/load operations
- Configuration override system

#### 4. **Training Pipeline Tests**
- Trainer initialization and setup
- Data loading pipeline validation
- Optimizer configuration
- Dry run functionality

#### 5. **Error Handling Tests**
- Invalid configuration handling
- Model loading error recovery
- Graceful failure scenarios

#### 6. **Integration Tests**
- End-to-end config workflow
- Model + Adapter + Config integration
- File operations and persistence

### 🌐 CI/CD Integration

Tests are automatically run on:
- ✅ **Python 3.8 & 3.11** (Ubuntu)
- ✅ **Push to main/develop branches**
- ✅ **Pull requests**
- ✅ **Multiple test types**: Basic + Comprehensive

### 📈 Test Results
```
22 passed, 6 warnings in ~9 seconds
```

**For detailed test coverage documentation, see: [TEST_COVERAGE.md](TEST_COVERAGE.md)**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v1.0.0
- Initial release
- Universal model wrapper
- Configuration-based training
- REST API
- CLI interface
- Multiple data format support
