# Train System Installation Guide

## Overview

Train System provides flexible installation options to suit different use cases:

- **Base Installation**: Core training functionality only
- **Web API Installation**: Includes web API server for remote training
- **Complete Installation**: All features and optional dependencies

## Installation Methods

### 1. Base Installation (Recommended for most users)

Install only the core training functionality:

```bash
pip install train-system
# or from source:
pip install -e .
```

**Includes:**
- Core training framework
- Configuration management
- CLI tools (`ts-train`, `ts-template`, etc.)
- Model adapters and wrappers

**Excludes:**
- Web API server
- Real-time web interface
- Advanced web deployment features

### 2. Web API Installation

Install with web API capabilities:

```bash
pip install train-system[web]
# or from source:
pip install -e ".[web]"
```

**Additional Features:**
- Basic API server (`ts-api`)
- Enhanced web API server (`ts-web-api`)
- Real-time training updates via WebSocket
- File upload and management
- Multi-job training management
- Web dashboard interface

**Additional Commands:**
- `ts-api` - Start basic API server
- `ts-web-api` - Start enhanced web API server
- `python -m train_system.api.web_server` - Enhanced API with options

### 3. Complete Installation

Install with all optional dependencies:

```bash
pip install train-system[all]
# or from source:
pip install -e ".[all]"
```

**Additional Features:**
- Web API (as above)
- Advanced model libraries (timm, transformers)
- Experiment tracking (wandb, mlflow)
- Enhanced visualization (matplotlib, seaborn)
- Advanced image processing (opencv)

### 4. Development Installation

For developers and contributors:

```bash
pip install train-system[dev]
# or from source:
pip install -e ".[dev]"
```

**Includes:**
- Testing frameworks (pytest)
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Pre-commit hooks

## Web API Dependencies

The web API functionality requires these additional packages:

- `flask>=2.0.0` - Web framework
- `flask-cors>=3.0.0` - Cross-origin resource sharing
- `flask-socketio>=5.0.0` - Real-time communication
- `requests>=2.25.0` - HTTP client
- `werkzeug>=2.0.0` - WSGI utilities

## Installation Verification

### Base Installation

```bash
# Check CLI availability
ts-train --help
ts-template --help

# Test core functionality
python -c "from train_system import UnifiedTrainer; print('✅ Base installation OK')"
```

### Web Installation

```bash
# Check web API availability
ts-web-api --help
python -m train_system.api.web_server --help

# Test web API functionality
python -c "from train_system.api.web_server import EnhancedTrainingAPI; print('✅ Web API OK')"
```

## Conditional Usage

The system gracefully handles missing optional dependencies:

```python
# This will work regardless of installation type
from train_system import UnifiedTrainer
from train_system.config import UnifiedTrainingConfig

# This will only work with web installation
try:
    from train_system.api.web_server import EnhancedTrainingAPI
    print("Web API available")
except ImportError:
    print("Web API not installed - use: pip install train-system[web]")
```

## Migration from Old Versions

If you have an older version with hardcoded API dependencies:

1. Uninstall old version: `pip uninstall train-system`
2. Install with desired features: `pip install train-system[web]`
3. Update any scripts that assumed API availability

## Docker Installation

Base image:
```dockerfile
FROM python:3.11-slim
RUN pip install train-system
```

Web API image:
```dockerfile
FROM python:3.11-slim
RUN pip install train-system[web]
EXPOSE 5000
CMD ["python", "-m", "train_system.api.web_server", "--host", "0.0.0.0"]
```

## Troubleshooting

### "ts-api command not found"

You need to install web dependencies:
```bash
pip install train-system[web]
```

### "ImportError: No module named 'flask'"

Install web dependencies:
```bash
pip install flask flask-cors flask-socketio requests werkzeug
# or
pip install train-system[web]
```

### Web API fails to start

1. Check if web dependencies are installed:
   ```python
   python -c "import flask; print('Flask available')"
   ```

2. Try basic installation test:
   ```python
   python -c "from train_system.api.web_server import EnhancedTrainingAPI; print('OK')"
   ```

3. Check port availability:
   ```bash
   netstat -an | grep :5000
   ```

## Support

- For installation issues: Check [Installation Guide](docs/INSTALLATION.md)
- For web API usage: Check [API Documentation](train_system/api/README.md)
- For general usage: Check [README.md](README.md)
