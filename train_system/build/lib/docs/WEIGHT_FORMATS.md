# Weight Format Examples for Train System

This document shows how to configure different weight formats for different model types.

## Example 1: YOLO Model with .pt Format

```yaml
model:
  name: "YOLO_Model"
  type: "file"
  path: "models/yolo_standalone.py"
  class_name: "YOLOStandalone"
  num_classes: 2

data:
  name: "URS_Dataset"
  type: "class_folders"
  train_path: "data/train"
  val_path: "data/val"
  real_path: "real"
  fake_path: "fake"
  class_mapping:
    real: 0
    fake: 1

training:
  epochs: 10
  learning_rate: 0.001
  optimizer: "adamw"

output:
  output_dir: "training_output"
  experiment_name: "yolo_experiment"
  weight_format: "pt"  # Use .pt for YOLO models
```

## Example 2: Standard PyTorch Model with .pth Format

```yaml
model:
  name: "ResNet_Model"
  type: "torchvision"
  model_name: "resnet18"
  num_classes: 2

data:
  name: "Image_Dataset"
  type: "folder"
  train_path: "data/train"
  val_path: "data/val"

training:
  epochs: 20
  learning_rate: 0.001
  optimizer: "adam"

output:
  output_dir: "training_output"
  experiment_name: "resnet_experiment"
  weight_format: "pth"  # Use .pth for standard PyTorch models
```

## Example 3: Auto Format Selection

```yaml
model:
  name: "Custom_Model"
  type: "file"
  path: "models/custom_model.py"
  class_name: "CustomModel"

data:
  name: "Custom_Dataset"
  type: "csv"
  train_path: "data/train.csv"
  val_path: "data/val.csv"

training:
  epochs: 15
  learning_rate: 0.0001

output:
  output_dir: "training_output"
  experiment_name: "custom_experiment"
  weight_format: "auto"  # Let train_system decide (defaults to .pth)
```

## Converting Between Formats

If you need to convert weights between formats, use the conversion utilities:

```bash
# Convert .pth to .pt
python train_system/weight_manager.py convert best_checkpoint.pth -f pt

# Convert .pt to .pth  
python train_system/weight_manager.py convert best_checkpoint.pt -f pth
```

## Best Practices

1. **For YOLO models**: Always use `weight_format: "pt"`
2. **For standard PyTorch models**: Use `weight_format: "pth"`
3. **When unsure**: Use `weight_format: "auto"` and convert later if needed
4. **For deployment**: Check what format your inference framework expects

## Loading Weights in Your Code

### For YOLO Models
```python
from models.yolo_standalone import YOLODetector

# Make sure to use .pt weights for YOLO
model = YOLODetector(model_path='training_output/yolo_experiment/best_checkpoint.pt')
```

### For Standard PyTorch Models
```python
import torch

# Load .pth weights normally
checkpoint = torch.load('training_output/resnet_experiment/best_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```
