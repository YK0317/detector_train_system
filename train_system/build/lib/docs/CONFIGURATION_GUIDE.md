# Train System Configuration Guide

This guide covers all configuration options for the train_system package.

## Overview

The train_system uses YAML configuration files to define:
- Model architecture and parameters
- Dataset configuration and paths
- Training hyperparameters
- Output settings

## Configuration Structure

```yaml
model:
  # Model configuration
data:
  # Dataset configuration
training:
  # Training parameters
output:
  # Output settings
device: "auto"
seed: 42
```

## Model Configuration

```yaml
model:
  name: "MyModel"
  type: "file"  # or "torchvision"
  path: "path/to/model.py"  # for type: "file"
  class_name: "ModelClass"  # for type: "file"
  model_name: "resnet18"    # for type: "torchvision"
  pretrained: true
  num_classes: 2
  img_size: 224
  
  # Optional adapter configuration
  adapter: "AutoAdapter"  # or specific adapter class
  adapter_config:
    custom_param: "value"
```

## Dataset Configuration

### Standard Dataset Types

#### Image Dataset
```yaml
data:
  name: "MyDataset"
  type: "image"
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"  # optional
  img_size: 224
  batch_size: 32
  num_workers: 4
```

#### CSV Dataset
```yaml
data:
  name: "MyCSVData"
  type: "csv"
  train_path: "data/train.csv"
  val_path: "data/val.csv"
  test_path: "data/test.csv"  # optional
  batch_size: 32
```

### Class Folders Configuration

The train_system supports three flexible patterns for class-based folder structures:

#### 1. Base Path + Relative Paths (Recommended)

For clean configurations with a base path and relative class folders:

```yaml
data:
  name: "ClassFolders"
  type: "class_folders"
  
  # Base paths for each split
  train_path: "C:\\Users\\Data\\train"
  val_path: "C:\\Users\\Data\\validation"
  test_path: "C:\\Users\\Data\\test"  # optional
  
  # Relative paths within the base paths
  real_path: "real"  # Creates train_path/real, val_path/real, etc.
  fake_path: "fake"  # Creates train_path/fake, val_path/fake, etc.
  
  class_mapping:
    real: 0
    fake: 1
  
  img_size: 224
  batch_size: 32
  num_workers: 4
```

**Directory Structure Created:**
```
C:\Users\Data\
├── train\
│   ├── real\      # Class 0 training images
│   └── fake\      # Class 1 training images
├── validation\
│   ├── real\      # Class 0 validation images
│   └── fake\      # Class 1 validation images
└── test\          # optional
    ├── real\      # Class 0 test images
    └── fake\      # Class 1 test images
```

#### 2. Absolute Paths (Legacy Support)

For explicit control over each path:

```yaml
data:
  type: "class_folders"
  
  # Training paths
  real_path: "C:\\full\\path\\to\\train\\real"
  fake_path: "C:\\full\\path\\to\\train\\fake"
  
  # Validation paths
  val_real_path: "C:\\full\\path\\to\\val\\real"
  val_fake_path: "C:\\full\\path\\to\\val\\fake"
  
  # Test paths (optional)
  test_real_path: "C:\\full\\path\\to\\test\\real"
  test_fake_path: "C:\\full\\path\\to\\test\\fake"
  
  class_mapping:
    real: 0
    fake: 1
```

#### 3. Class Folders Dictionary (Full Control)

For complex configurations with multiple classes and custom splits:

```yaml
data:
  type: "class_folders"
  class_folders:
    train:
      real: "data/train/real"
      fake: "data/train/fake"
      other: "data/train/other"
    val:
      real: "data/val/real"
      fake: "data/val/fake"
      other: "data/val/other"
    test:  # optional
      real: "data/test/real"
      fake: "data/test/fake"
      other: "data/test/other"
  
  class_mapping:
    real: 0
    fake: 1
    other: 2
```

### Dataset Configuration Options

```yaml
data:
  img_size: 224           # Image resize dimension
  batch_size: 32          # Batch size for training
  num_workers: 4          # DataLoader workers
  max_samples: null       # Limit samples per class (for testing)
  shuffle: true           # Shuffle training data
  pin_memory: true        # Pin memory for faster GPU transfer
  
  # Data augmentation
  augmentation:
    horizontal_flip: 0.5
    rotation: 15
    brightness: 0.2
```

## Training Configuration

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adamw"      # adamw, adam, sgd
  weight_decay: 0.01
  momentum: 0.9           # for SGD
  
  # Learning rate scheduling
  scheduler: "cosine"     # cosine, step, exponential, plateau
  step_size: 30           # for step scheduler
  gamma: 0.1             # for step/exponential schedulers
  
  # Training settings
  val_frequency: 1        # Validate every N epochs
  save_frequency: 5       # Save checkpoint every N epochs
  log_interval: 10        # Log every N batches
  early_stopping: 10      # Stop after N epochs without improvement
  
  # Loss function
  criterion: "cross_entropy"  # cross_entropy, bce, mse
  
  # Mixed precision training
  mixed_precision: true
```

## Output Configuration

```yaml
output:
  output_dir: "training_output"
  experiment_name: "my_experiment"
  save_best_only: false
  save_last: true
  tensorboard: true
  save_predictions: false
  
  # Weight file format configuration
  weight_format: "pth"  # "auto", "pth", "pt"
```

### Weight Format Options

- **`pth`** (default): Standard PyTorch format, compatible with most PyTorch models
- **`pt`**: Alternative PyTorch format, required for YOLO models and some other frameworks
- **`auto`**: Automatically choose format (defaults to `pth`)

**When to use `.pt` format:**
- YOLO models (YOLOv5, YOLOv8, Ultralytics)
- Some pre-trained models that expect `.pt` extension
- When integrating with frameworks that specifically require `.pt`

**When to use `.pth` format:**
- Most standard PyTorch models
- Custom models and research projects
- General PyTorch training workflows

## Global Settings

```yaml
device: "auto"           # auto, cpu, cuda, cuda:0
seed: 42                 # Random seed for reproducibility
description: "Experiment description"
tags: ["tag1", "tag2"]   # Experiment tags
```

## Configuration Templates

Generate configuration templates:

```bash
# Generate basic template
train-system generate-config basic_template.yaml

# Generate with specific model type
train-system generate-config --model-type torchvision template.yaml

# Generate with class folders
train-system generate-config --data-type class_folders template.yaml
```

## Configuration Validation

The system validates configurations and provides helpful error messages:

- Path existence checking
- Parameter type validation
- Required field verification
- Value range checking

## Examples

See the `examples/` directory for complete configuration examples:
- `basic_config.yaml` - Simple image classification
- `class_folders_config.yaml` - Class-based folder structure
- `flexible_paths_config.yaml` - Base path + relative paths
- `advanced_config.yaml` - Advanced features and settings

## Environment Variables

Override configuration values with environment variables:

```bash
export TRAIN_SYSTEM_OUTPUT_DIR="custom_output"
export TRAIN_SYSTEM_DEVICE="cuda:1"
export TRAIN_SYSTEM_BATCH_SIZE="16"
```

## Best Practices

1. **Use relative paths** when possible for portability
2. **Set appropriate batch sizes** based on your GPU memory
3. **Use validation frequency** to balance training speed and monitoring
4. **Set random seeds** for reproducible experiments
5. **Use descriptive experiment names** for easy identification
6. **Tag experiments** for better organization
