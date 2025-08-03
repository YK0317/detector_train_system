# Train System Enhanced Features

## Adapter Specification from Config

You can now specify the adapter type and configuration directly in your config file:

```yaml
model:
  name: "MyModel"
  type: "file"
  path: "models/my_model.py"
  class_name: "MyModelClass"
  
  # Adapter specification
  adapter: "classification"  # or "detection", "segmentation", "auto"
  adapter_config:
    num_classes: 2
    use_softmax: true
    threshold: 0.5
```

### Available Adapters

1. **"auto"** - AutoAdapter that automatically detects output format
2. **"classification"** - For classification models
3. **"detection"** - For object detection models (YOLO, etc.)
4. **"segmentation"** - For segmentation models
5. **Custom adapter class** - Specify full class path like "my_package.CustomAdapter"

### Adapter Configuration

Each adapter can accept specific configuration parameters:

- **Classification Adapter**:
  ```yaml
  adapter_config:
    num_classes: 2
    use_softmax: true
    threshold: 0.5
  ```

- **Detection Adapter**:
  ```yaml
  adapter_config:
    num_classes: 80
    confidence_threshold: 0.25
    nms_threshold: 0.45
    max_detections: 100
  ```

## Class Folder Specification

You can now specify separate folders for different classes:

### Method 1: Direct real/fake paths
```yaml
data:
  type: "class_folders"
  real_path: "data/real_images"
  fake_path: "data/fake_images"
  class_mapping:
    real: 0
    fake: 1
```

### Method 2: Multiple class folders
```yaml
data:
  type: "class_folders"
  class_folders:
    real: "data/real_images"
    fake: "data/fake_images"
    synthetic: "data/synthetic_images"
  class_mapping:
    real: 0
    fake: 1
    synthetic: 2
```

### Method 3: Mixed approach
```yaml
data:
  type: "class_folders"
  real_path: "data/real_images"
  fake_path: "data/fake_images"
  class_folders:
    extra_class: "data/extra_images"
  class_mapping:
    real: 0
    fake: 1
    extra_class: 2
```

## Validation Data Handling

For `class_folders` type, the system automatically looks for validation data in these patterns:

1. `{parent_folder}/validation/{class_name}/`
2. Uses `val_path` with class subfolders
3. Falls back to training folders if validation not found

## Examples

### Example 1: YOLO with Classification Adapter
```yaml
model:
  name: "YOLO_Classification"
  type: "file"
  path: "../models/yolo_standalone.py"
  class_name: "YOLOStandalone"
  adapter: "classification"
  adapter_config:
    num_classes: 2

data:
  type: "class_folders"
  real_path: "C:\\data\\real"
  fake_path: "C:\\data\\fake"
```

### Example 2: Custom Model with Detection Adapter
```yaml
model:
  name: "CustomDetector"
  type: "file"
  path: "models/detector.py"
  class_name: "MyDetector"
  adapter: "detection"
  adapter_config:
    confidence_threshold: 0.3
    nms_threshold: 0.5

data:
  type: "folder"
  train_path: "data/train"
  val_path: "data/val"
```

### Example 3: Multiple Class Folders
```yaml
model:
  adapter: "classification"
  adapter_config:
    num_classes: 3

data:
  type: "class_folders"
  class_folders:
    real: "dataset/real"
    fake: "dataset/fake"
    synthetic: "dataset/synthetic"
  class_mapping:
    real: 0
    fake: 1
    synthetic: 2
```

## Generate Templates

Use the template generator to create example configurations:

```bash
python generate_templates.py
```

This creates:
- `class_folders_template.yaml` - Example with class folders
- `yolo_template.yaml` - YOLO with detection adapter
- `blip_template.yaml` - BLIP model example
- `torchvision_template.yaml` - TorchVision models
- `generic_template.yaml` - Generic template

## Usage

```bash
# Train with class folders
train-system train class_folders_template.yaml

# Train YOLO with detection adapter
train-system train yolo_template.yaml

# Train with custom configuration
train-system train my_custom_config.yaml
```
