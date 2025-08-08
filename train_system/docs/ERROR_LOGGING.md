# Error Logging Documentation

## Overview

The train-system now includes comprehensive error logging functionality that automatically creates detailed error log files whenever training fails to initialize or encounters problems during execution.

## Features

### 🔍 **Automatic Error Detection**
- **Initialization Errors**: Configuration problems, dependency issues
- **Model Loading Errors**: Invalid model paths, architecture issues, missing dependencies
- **Data Loading Errors**: Invalid dataset paths, format issues, memory problems
- **Optimizer Setup Errors**: Invalid optimizer names, parameter issues
- **Training Errors**: Runtime errors during training loops
- **Checkpoint Loading Errors**: Invalid checkpoint files, compatibility issues
- **External Trainer Errors**: Issues with custom training implementations

### 📝 **Detailed Error Logs**

Error logs are automatically saved to the output directory with the following information:

#### **File Location**
```
{output_dir}/{experiment_name}/error_log_{phase}_{timestamp}.log
```

#### **Log Contents**
1. **Error Summary**
   - Error phase (initialization, model_loading, etc.)
   - Error type and message
   - Timestamp

2. **Full Traceback**
   - Complete Python stack trace
   - Detailed error context

3. **System Information**
   - Platform and Python version
   - PyTorch and CUDA versions
   - GPU information and memory usage
   - System memory status

4. **Configuration Details**
   - Complete training configuration
   - Model and data settings
   - Training parameters

5. **Phase-Specific Troubleshooting Tips**
   - Common solutions for the error type
   - Recommended next steps
   - Configuration suggestions

### 🚨 **Error Phases**

The system categorizes errors into specific phases for better troubleshooting:

- **`initialization`**: Basic setup and configuration validation
- **`model_loading`**: Model creation and weight loading
- **`data_loading`**: Dataset loading and data loader setup  
- **`optimizer_setup`**: Optimizer and scheduler configuration
- **`training`**: Runtime errors during training loops
- **`checkpoint_loading`**: Resume training and checkpoint issues
- **`external_trainer`**: Custom trainer implementation errors

## Usage Examples

### 🔧 **Basic Usage**

Error logging is automatically enabled - no configuration needed:

```yaml
# config.yaml
model:
  type: "file"
  path: "/invalid/path/model.py"  # This will trigger model_loading error
  
output:
  output_dir: "training_output"
  experiment_name: "my_experiment"
```

When training fails, you'll see:
```bash
❌ TRAINING FAILED - MODEL_LOADING
📝 Error log saved to: training_output/my_experiment/error_log_model_loading_20250808_143022.log
🔍 Error: FileNotFoundError: Model file not found: /invalid/path/model.py
```

### 📋 **CLI Integration**

The CLI automatically shows error log locations:

```bash
# Run training
python -m train_system.cli.main train config.yaml

# Output on error:
❌ Training failed: Model file not found: /invalid/path/model.py
📁 Output directory: training_output/my_experiment  
📝 Detailed error log should be available in the output directory
🔍 Latest error log: training_output/my_experiment/error_log_model_loading_20250808_143022.log
```

### 🔍 **Debug Mode**

Enable full traceback output:

```bash
# Windows
set TRAIN_SYSTEM_DEBUG=1
python -m train_system.cli.main train config.yaml

# Linux/Mac
TRAIN_SYSTEM_DEBUG=1 python -m train_system.cli.main train config.yaml
```

## Error Log Format

### 📄 **Sample Error Log**

```
================================================================================
TRAINING ERROR LOG - 2025-08-08T14:30:22.123456
================================================================================

ERROR PHASE: model_loading
ERROR TYPE: FileNotFoundError
ERROR MESSAGE: Model file not found: /invalid/path/model.py

FULL TRACEBACK:
----------------------------------------
Traceback (most recent call last):
  File "train_system/core/trainer.py", line 456, in _load_file_model
    raise FileNotFoundError(f"Model file not found: {model_path}")
FileNotFoundError: Model file not found: /invalid/path/model.py
----------------------------------------

SYSTEM INFORMATION:
----------------------------------------
platform: Windows-10-10.0.19041-SP0
python_version: 3.9.7
pytorch_version: 1.12.0
cuda_available: true
cuda_version: 11.6
gpu_count: 1
memory_total_gb: 16.0
memory_available_gb: 8.5

DEVICE INFORMATION:
----------------------------------------
device: cuda:0
gpu_name: NVIDIA GeForce RTX 3060
gpu_memory_total: 12884901888
gpu_memory_allocated: 0
gpu_memory_cached: 0

CONFIGURATION:
----------------------------------------
{
  "model": {
    "type": "file",
    "path": "/invalid/path/model.py",
    "class_name": "CustomModel",
    "num_classes": 2
  },
  "data": {
    "train_dir": "data/train",
    "val_dir": "data/val",
    "batch_size": 32
  },
  "training": {
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },
  "output": {
    "output_dir": "training_output",
    "experiment_name": "my_experiment"
  }
}

TROUBLESHOOTING TIPS:
----------------------------------------
• Verify model path exists and is accessible
• Check model file format and compatibility
• Ensure model architecture is supported
• Verify model class name and module path for custom models
• Check if required model dependencies are installed (timm, torchvision, etc.)

COMMON SOLUTIONS:
• Update PyTorch and dependencies: pip install --upgrade torch torchvision
• Clear Python cache: python -c 'import torch; torch.hub._get_cache_dir()' then delete cache
• Check available disk space in output directory
• Restart Python session to clear any memory issues
• Try with a smaller batch_size or simpler model first

================================================================================
END OF ERROR LOG
================================================================================
```

## Fallback Behavior

### 🛡️ **Robust Error Handling**

If the main error logging fails (e.g., permission issues), the system:

1. **Creates fallback log** in current directory
2. **Prints error to console** with basic information
3. **Never crashes** due to logging failures
4. **Preserves original error** for debugging

### 📁 **Output Directory Handling**

- **Auto-creates** output directories if they don't exist
- **Uses fallback location** if configured directory is inaccessible
- **Handles permission issues** gracefully
- **Preserves logs** across multiple training attempts

## Integration with Existing Features

### 🔗 **Web API Integration**

Error logs are accessible through the web API:

```python
# Get latest error log for an experiment
GET /api/v1/experiments/{experiment_name}/error-logs

# Download specific error log
GET /api/v1/experiments/{experiment_name}/error-logs/{log_id}/download
```

### 📊 **TensorBoard Integration**

Error information is also logged to TensorBoard when available:

- Error summaries in scalar plots
- Configuration snapshots
- System information in text logs

### 🔄 **Checkpoint Integration**

When training is interrupted:

- **Interruption checkpoint** is saved automatically
- **Error log includes** checkpoint information
- **Resume instructions** are provided in the log

## Best Practices

### ✅ **Configuration Recommendations**

```yaml
output:
  output_dir: "/absolute/path/to/output"  # Use absolute paths
  experiment_name: "descriptive_name"    # Use descriptive names
  save_config: true                      # Always save config for debugging
```

### 🔍 **Debugging Workflow**

1. **Check console output** for immediate error information
2. **Locate error log** in output directory  
3. **Review troubleshooting tips** in the log
4. **Check system information** for compatibility issues
5. **Examine configuration** for parameter errors
6. **Try suggested solutions** from the log

### 📈 **Monitoring and Alerts**

For production use, consider:

- **Log monitoring tools** to track error patterns
- **Automated alerts** for training failures
- **Log aggregation** across multiple experiments
- **Error trend analysis** for system optimization

## API Reference

### 🛠️ **UnifiedTrainer Methods**

```python
def _create_error_log(self, error: Exception, phase: str, additional_info: dict = None) -> Path:
    """
    Create detailed error log file in output directory
    
    Args:
        error: The exception that occurred
        phase: Training phase where error occurred
        additional_info: Additional context information
        
    Returns:
        Path to created error log file
    """
```

### 📝 **Error Phases Constants**

```python
ERROR_PHASES = {
    "initialization": "Basic setup and configuration validation",
    "model_loading": "Model creation and weight loading", 
    "data_loading": "Dataset loading and data loader setup",
    "optimizer_setup": "Optimizer and scheduler configuration",
    "training": "Runtime errors during training loops",
    "checkpoint_loading": "Resume training and checkpoint issues",
    "external_trainer": "Custom trainer implementation errors"
}
```

## Troubleshooting

### ❓ **Common Issues**

**Q: Error logs are not being created**
- Check output directory permissions
- Verify disk space availability
- Check for filesystem limitations

**Q: Error logs are empty or corrupted**
- Check system memory availability
- Verify Python environment integrity
- Check for antivirus interference

**Q: Cannot read error logs**
- Check file encoding (should be UTF-8)
- Verify file permissions
- Check for file locking issues

### 🆘 **Getting Help**

If you encounter issues with error logging:

1. Check the console output for fallback error information
2. Verify system requirements and dependencies
3. Try running with `TRAIN_SYSTEM_DEBUG=1` for verbose output
4. Check GitHub issues for similar problems
5. Create a new issue with error log contents (if available)
