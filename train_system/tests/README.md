# Train System Test Suite

This directory contains comprehensive tests for the Train System, designed to work in CI/CD environments without requiring actual training data or heavy computations.

## Test Files

### 1. `test_comprehensive.py`
The main comprehensive test suite that covers:

- **System Imports**: Core component imports and initialization
- **Model Wrapping**: Model wrapping and adapter functionality
- **Configuration**: Configuration management and validation
- **Dataset Handling**: Dataset configuration parsing
- **Training Pipeline**: Training setup and dry runs
- **Utilities**: Helper functions and utilities
- **Error Handling**: Edge cases and error scenarios
- **Integration**: End-to-end workflow testing

### 2. `test_basic.py`
Basic functionality tests (existing file) for essential operations.

### 3. Test Configuration Files

- `pytest.ini`: PyTest configuration for proper test discovery
- `requirements-test.txt`: Minimal dependencies for testing
- `run_tests.py`: Standalone test runner (no pytest required)

## Running Tests

### Option 1: With PyTest (Recommended)
```bash
cd train_system
pip install -r requirements-test.txt
pytest tests/test_comprehensive.py -v
```

### Option 2: Standalone Test Runner
```bash
cd train_system
python run_tests.py                    # Run all tests
python run_tests.py --quick            # Run only basic tests
python run_tests.py --install-check    # Check installation only
```

### Option 3: Direct Execution
```bash
cd train_system
python tests/test_comprehensive.py     # Run comprehensive tests
python tests/test_basic.py             # Run basic tests
```

## CI/CD Integration

### GitHub Actions
The test suite is integrated with GitHub Actions:

- **Main CI** (`.github/workflows/ci.yaml`): Full CI pipeline with linting
- **Train System Tests** (`.github/workflows/train-system-tests.yml`): Focused testing

### Test Matrix
Tests run on multiple Python versions (3.8, 3.9, 3.10, 3.11) and include:

- Basic functionality tests
- Comprehensive integration tests
- Configuration validation
- Model creation and prediction
- Import compatibility checks

## Test Features

### 🔧 Mock-Based Testing
Tests use mocking to avoid requiring:
- Actual training data
- GPU resources
- External dependencies
- Network connections

### 🚀 Fast Execution
- Minimal dependencies
- No actual training
- Lightweight model operations
- Quick validation checks

### 🛡️ Error Resilience
Tests handle missing dependencies gracefully:
- Optional imports (Flask, etc.)
- Missing test data
- Configuration errors
- Platform differences

### 📊 Comprehensive Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflows
- **Configuration Tests**: YAML/JSON handling
- **Model Tests**: PyTorch model operations
- **CLI Tests**: Command-line interface

## Test Categories

### Core Functionality
- Model wrapping and adapters
- Configuration management
- Training pipeline setup

### System Integration
- Complete workflows
- File I/O operations
- Error handling

### Compatibility
- Python version compatibility
- Dependency handling
- Import systems

## Adding New Tests

### For New Features
1. Add test methods to appropriate test class in `test_comprehensive.py`
2. Follow naming convention: `test_feature_name`
3. Include docstrings explaining what's being tested
4. Use mocking for external dependencies

### Test Structure
```python
class TestNewFeature:
    """Test new feature functionality"""
    
    def test_basic_operation(self):
        """Test basic operation of new feature"""
        # Arrange
        setup_test_data()
        
        # Act
        result = new_feature_function()
        
        # Assert
        assert result is not None
        assert result.property == expected_value
```

### Best Practices
- Keep tests independent
- Use descriptive test names
- Mock external dependencies
- Test both success and failure cases
- Include edge cases

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure train_system is installed
cd train_system
pip install -e .

# Or add to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Missing Dependencies
```bash
# Install test requirements
pip install -r requirements-test.txt

# Or minimal requirements
pip install torch torchvision pytest
```

#### Test Failures
```bash
# Run with verbose output
python run_tests.py --verbose

# Run only basic tests
python run_tests.py --quick

# Check installation
python run_tests.py --install-check
```

### Debug Mode
```bash
# Run with Python debugger
python -m pytest tests/test_comprehensive.py -v -s --pdb
```

## Performance

### Test Timing
- Basic tests: ~10-30 seconds
- Comprehensive tests: ~1-3 minutes
- Full CI pipeline: ~5-10 minutes

### Resource Usage
- Minimal memory footprint
- CPU-only operations
- No GPU requirements
- No large file downloads

## Integration with IDEs

### VS Code
Tests auto-discovered with Python extension:
1. Open Command Palette (Ctrl+Shift+P)
2. "Python: Configure Tests"
3. Select "pytest"
4. Tests appear in Test Explorer

### PyCharm
1. Right-click on `tests` directory
2. "Run pytest in tests"
3. Or configure run configuration for specific test files

## Contributing

When adding new functionality to Train System:

1. **Add corresponding tests** in `test_comprehensive.py`
2. **Update test documentation** if needed
3. **Ensure tests pass** in CI pipeline
4. **Test on multiple Python versions** if possible

The test suite is designed to be maintainable, fast, and comprehensive while avoiding heavy dependencies or actual training operations.
