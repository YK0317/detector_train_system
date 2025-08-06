# 🔍 Detector Train System

A comprehensive deepfake detection and training system with unified model training capabilities and extensive test coverage.

## 📁 Repository Structure

```
detector_train_system/
├── 🧪 train_system/          # Universal PyTorch training system
├── 🤖 models/                # Standalone detection models
├── 📊 Capsule-Forensics-v2/  # Capsule network implementation
├── ⚙️ .github/workflows/     # CI/CD pipelines
└── 📋 config files           # Various training configurations
```

## 🎯 Main Components

### 🚀 Train System
**Location**: `train_system/`

A universal PyTorch training system that provides:
- **Universal Model Support**: Train any PyTorch model with unified interface
- **Configuration-Based Training**: YAML/JSON config files
- **REST API & CLI**: Remote training control and command-line interface
- **Adapter System**: Handles different model output formats automatically
- **Comprehensive Testing**: 22 test cases with 95%+ coverage

**Quick Start**:
```bash
cd train_system
pip install -e .
train-system create-template torchvision my_config.yaml
train-system train my_config.yaml
```

### 🤖 Detection Models
**Location**: `models/`

Standalone deepfake detection models:
- Xception
- EfficientNetB4  
- Meso4/MesoInception
- UCF models
- YOLO detection
- Capsule networks
- BLIP-based detection

### 📊 Capsule Forensics
**Location**: `Capsule-Forensics-v2/`

Advanced capsule network implementation for deepfake detection with specialized training scripts for various datasets.

## 🧪 Testing & CI

### Test Coverage
The system includes comprehensive testing with **22 test cases**:

| Component | Coverage | Status |
|-----------|----------|--------|
| Model Wrapping | 100% | ✅ |
| Configuration | 100% | ✅ |
| Training Pipeline | 90% | ✅ |
| Error Handling | 95% | ✅ |
| Integration | 100% | ✅ |

### CI/CD Pipeline
- ✅ **Automated Testing**: GitHub Actions on push/PR
- ✅ **Multi-Python**: Tests on Python 3.8 & 3.11
- ✅ **Cross-platform**: Ubuntu environment
- ✅ **Coverage Reports**: Detailed test coverage analysis

**Test Execution**:
```bash
cd train_system
python run_tests.py  # Quick test run
python -m pytest tests/test_comprehensive.py -v  # Detailed testing
```

## 🔧 Setup & Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA (optional, for GPU training)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd detector_train_system

# Install train system
cd train_system
pip install -e .

# Install detection models dependencies
pip install -r requirements.txt
```

## 📋 Configuration Examples

### Basic Training Config
```yaml
model:
  name: "ResNet18"
  type: "torchvision"
  architecture: "resnet18"
  pretrained: true
  num_classes: 2

data:
  type: "class_folders"
  train_path: "data/train"
  val_path: "data/val"
  
training:
  epochs: 10
  learning_rate: 0.001
  batch_size: 32
```

### Resume Training Config
```yaml
model:
  name: "UCF_Model"
  type: "file"
  path: "models/ucf_standalone.py"
  
training:
  resume_from: "checkpoint_epoch_5.pth"
  epochs: 15
```

## 🚀 Quick Start Examples

### 1. Train a TorchVision Model
```bash
cd train_system
train-system create-template torchvision resnet_config.yaml
# Edit config file as needed
train-system train resnet_config.yaml
```

### 2. Use Standalone Detection Model
```python
from models.xception_standalone import XceptionModel

model = XceptionModel(num_classes=2)
model.load_checkpoint("checkpoint_epoch_5.pth")
predictions = model.predict(image_tensor)
```

### 3. Run Comprehensive Tests
```bash
cd train_system
python run_tests.py
# Output: 22 passed, 6 warnings in ~9 seconds ✅
```

## 📊 Features

### 🔄 Universal Training System
- Any PyTorch model compatibility
- Automatic adapter detection
- Configuration-based training
- Resume training capabilities
- Memory optimization

### 🧪 Robust Testing
- 22 comprehensive test cases
- Cross-platform compatibility  
- CI/CD integration
- Error handling validation
- Performance benchmarking

### 🤖 Detection Models
- Multiple deepfake detection architectures
- Pre-trained weights available
- Standalone inference scripts
- Specialized dataset handlers

### ⚙️ Advanced Configuration
- Template system for quick setup
- Override mechanisms
- Validation and error checking
- YAML/JSON support

## 📈 Performance

- **Training Speed**: Optimized for GPU acceleration
- **Memory Usage**: Advanced memory management
- **Test Speed**: 22 tests in ~9 seconds
- **Model Loading**: Fast checkpoint resume
- **API Response**: Sub-second response times

## 🛠️ Development

### Running Tests
```bash
# All tests
cd train_system && python run_tests.py

# Specific test category
python -m pytest tests/test_comprehensive.py::TestModelWrapping -v

# With coverage
python -m pytest tests/test_comprehensive.py --cov=train_system
```

### Adding New Models
1. Create model file in `models/`
2. Implement standard interface
3. Add configuration template
4. Update tests
5. Submit PR

### Contributing
1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📚 Documentation

- **[Train System README](train_system/README.md)** - Detailed train system docs
- **[Test Coverage](train_system/TEST_COVERAGE.md)** - Comprehensive test documentation
- **[Performance Guide](train_system/PERFORMANCE_OPTIMIZATIONS.md)** - Optimization tips
- **[Adapter Guide](train_system/ADAPTER_GUIDE.md)** - Custom adapter development

## 🚦 CI Status

![Train System Tests](https://github.com/YK0317/detector_train_system/workflows/Train%20System%20Tests/badge.svg)

**Latest Test Results**: ✅ 22 passed, 6 warnings  
**Coverage**: 95%+ across all components  
**Platform**: Ubuntu, Python 3.8 & 3.11

## 📄 License

MIT License - see LICENSE file for details.

---

*Last Updated: August 6, 2025*  
*Repository Version: 1.0*  
*Test Suite: 22 comprehensive tests*
