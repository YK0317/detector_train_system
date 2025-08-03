# Adapter Specification Guide

## Overview
The train_system now supports optional adapter specification in configuration files. If no adapter is specified, it defaults to AutoAdapter for automatic detection.

## Usage

### Default Behavior (AutoAdapter)
```yaml
model:
  name: "MyModel"
  type: "file"
  path: "models/my_model.py"
  class_name: "MyModelClass"
  # No adapter specified - uses AutoAdapter by default
```

### Explicit Adapter Specification
```yaml
model:
  name: "MyModel"
  type: "file"
  path: "models/my_model.py"
  class_name: "MyModelClass"
  
  # Specify adapter type
  adapter: "standard"  # or "auto", "logits_features", "dict", etc.
  adapter_config: {}   # Optional configuration for the adapter
```

## Available Adapters

1. **"auto"** - AutoAdapter (automatic detection) - Default
2. **"standard"** - StandardAdapter (models returning only logits)
3. **"logits_features"** - LogitsAndFeaturesAdapter (models returning (logits, features))
4. **"dict"** - DictOutputAdapter (models returning dictionary outputs)
5. **"capsule"** - CapsuleNetworkAdapter (specialized for CapsuleNet)
6. **"ucf"** - UCFAdapter (specialized for UCF models)

## Examples

### Example 1: Default AutoAdapter
```yaml
model:
  name: "YOLO"
  type: "file"
  path: "../models/yolo_standalone.py"
  class_name: "YOLOStandalone"
  # AutoAdapter will be used automatically
```

### Example 2: Explicit Standard Adapter
```yaml
model:
  name: "ResNet"
  type: "torchvision"
  architecture: "resnet18"
  adapter: "standard"
  adapter_config: {}
```

### Example 3: Custom Adapter with Configuration
```yaml
model:
  name: "CustomModel"
  type: "file"
  path: "models/custom.py"
  class_name: "CustomModelClass"
  adapter: "dict"
  adapter_config:
    output_key: "predictions"
    features_key: "features"
```

## Benefits

- **Automatic**: Works out of the box with AutoAdapter
- **Flexible**: Allows manual adapter specification when needed
- **Backward Compatible**: Existing configs continue to work
- **Configurable**: Adapters can accept configuration parameters

## Testing

Use these test configurations:
- `test_default_adapter.yaml` - Tests AutoAdapter (default)
- `test_adapter_config.yaml` - Tests explicit adapter specification

```bash
# Test default behavior
train-system train test_default_adapter.yaml

# Test explicit adapter
train-system train test_adapter_config.yaml
```
