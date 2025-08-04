# Train-System on Google Colab

This guide helps you install and use the train-system package on Google Colab.

## Quick Installation

### Method 1: Automatic Installation (Recommended)

```python
# In a Colab cell, run:
!git clone <your-repo-url>
%cd train_system
!python install_colab.py
```

### Method 2: Manual Installation

```python
# Install dependencies first
!pip install -r requirements-colab.txt

# Clean any existing installations
!pip uninstall train-system -y

# Install in development mode
!pip install -e . --verbose
```

### Method 3: Minimal Installation

```python
# If you encounter build errors, try this minimal approach:
!pip install --upgrade pip setuptools wheel
!pip install -e . --no-deps --force-reinstall
!pip install torch torchvision PyYAML tqdm numpy pandas tensorboard
```

## Common Issues and Solutions

### Issue 1: OpenCV Conflicts
```python
# Remove conflicting OpenCV packages
!pip uninstall opencv-python opencv-python-headless -y
!pip install opencv-python-headless==4.8.1.78
```

### Issue 2: Build Errors
```python
# Clean build artifacts
import shutil, os
for path in ['build', 'dist', '*.egg-info']:
    if os.path.exists(path):
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
```

### Issue 3: Memory Issues
```python
# Restart runtime if needed
exit()  # This will restart the Colab runtime
```

## Testing Installation

```python
# Test import
import train_system
print("✅ Train-system imported successfully!")

# Test dry-run
!python -m train_system.cli.main dry-run examples/basic_config.yaml
```

## Example Usage

```python
# 1. Create your config file
config = {
    'model': {
        'type': 'custom',
        'name': 'your_model'
    },
    'data': {
        'type': 'file_based',
        'train_path': '/content/data/train',
        'val_path': '/content/data/val'
    },
    'training': {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

import yaml
with open('my_config.yaml', 'w') as f:
    yaml.dump(config, f)

# 2. Run training
!python -m train_system.cli.main train my_config.yaml
```

## GPU Setup for Colab

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## File Management in Colab

```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Copy your data to Colab
!cp -r /content/drive/MyDrive/your_data /content/data
```

## Tips for Success

1. **Always restart runtime** after installing packages if you encounter import errors
2. **Use absolute paths** for data and model files
3. **Save checkpoints** to Google Drive for persistence
4. **Monitor GPU memory** usage during training
5. **Use smaller batch sizes** if you run out of memory

## Troubleshooting Commands

```python
# Check package installation
!pip list | grep train-system

# Verify file structure
!find . -name "*.py" | head -20

# Check Python environment
import sys
print(sys.executable)
print(sys.version)

# Memory usage
!free -h
!nvidia-smi  # If using GPU
```
