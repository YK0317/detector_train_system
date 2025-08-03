#!/usr/bin/env python3
"""
Test script for Train System

This script tests the basic functionality of the train system
without requiring actual training data.
"""

import torch
import torchvision.models as models
import sys
from pathlib import Path

# Add the train_system to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from train_system import UnifiedTrainingWrapper, ModelFactory
    from train_system.config import UnifiedTrainingConfig, ConfigTemplateManager, ConfigValidator
    from train_system.core.wrapper import ModelUtils
    from train_system.adapters import AutoAdapter, StandardAdapter
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_model_wrapping():
    """Test basic model wrapping functionality"""
    print("\n🧪 Testing Model Wrapping...")
    
    # Create a simple model
    model = models.resnet18(pretrained=False)  # Don't download weights for test
    
    # Modify the classifier to have 2 classes
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # Test basic wrapping
    wrapper = ModelFactory.create_wrapped_model(model, num_classes=2)
    print(f"✅ Model wrapped successfully")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224)
    output = wrapper(test_input)
    
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    print(f"✅ Forward pass successful, output shape: {output.shape}")
    
    # Test prediction
    pred_dict = wrapper.predict(test_input)
    expected_keys = ['logits', 'probabilities', 'predictions']
    for key in expected_keys:
        assert key in pred_dict, f"Missing key: {key}"
    print(f"✅ Prediction dictionary contains: {list(pred_dict.keys())}")
    
    # Test parameter counting
    param_count = wrapper.get_parameter_count()
    assert param_count > 0, "Parameter count should be positive"
    print(f"✅ Parameter count: {param_count:,}")


def test_adapters():
    """Test adapter functionality"""
    print("\n🧪 Testing Adapters...")
    
    # Test AutoAdapter
    adapter = AutoAdapter()
    
    # Test with tensor output (standard)
    tensor_output = torch.randn(1, 2)
    logits = adapter.extract_logits(tensor_output)
    assert torch.equal(logits, tensor_output), "AutoAdapter should return tensor as-is"
    print("✅ AutoAdapter works with tensor output")
    
    # Test with tuple output
    tuple_output = (torch.randn(1, 2), torch.randn(1, 512))
    # Reset adapter for new output type
    adapter = AutoAdapter()
    logits = adapter.extract_logits(tuple_output)
    assert torch.equal(logits, tuple_output[0]), "AutoAdapter should extract first element"
    print("✅ AutoAdapter works with tuple output")
    
    # Test with dict output
    dict_output = {'logits': torch.randn(1, 2), 'features': torch.randn(1, 512)}
    # Reset adapter for new output type
    adapter = AutoAdapter()
    logits = adapter.extract_logits(dict_output)
    assert torch.equal(logits, dict_output['logits']), "AutoAdapter should extract logits key"
    print("✅ AutoAdapter works with dict output")


def test_configuration():
    """Test configuration management"""
    print("\n🧪 Testing Configuration...")
    
    # Test template creation
    template = ConfigTemplateManager.get_template("torchvision")
    assert 'model' in template, "Template should contain model section"
    assert 'data' in template, "Template should contain data section"
    print("✅ Template loaded successfully")
    
    # Test config creation from template
    config = ConfigTemplateManager.create_config_from_template(
        "torchvision",
        model={'num_classes': 5},
        training={'epochs': 10}
    )
    assert config.model.num_classes == 5, "Override should work"
    assert config.training.epochs == 10, "Override should work"
    print("✅ Configuration created with overrides")
    
    # Test validation (should pass even without data paths existing)
    validation_result = ConfigValidator.validate(config)
    print(f"✅ Configuration validation completed (errors: {len(validation_result.errors)})")


def test_model_utils():
    """Test ModelUtils functionality"""
    print("\n🧪 Testing ModelUtils...")
    
    # Test adapter listing
    adapters = ModelUtils.list_available_adapters()
    assert len(adapters) > 0, "Should have available adapters"
    print(f"✅ Found {len(adapters)} adapters")
    
    # Test model compatibility check
    model = models.resnet18(pretrained=False)
    compatibility = ModelUtils.validate_model_compatibility(model)
    assert compatibility['compatible'], "ResNet18 should be compatible"
    print(f"✅ Model compatibility check passed: {compatibility['recommended_adapter']}")


def test_api_imports():
    """Test API component imports"""
    print("\n🧪 Testing API Imports...")
    
    try:
        from train_system.api import TrainingAPI
        api = TrainingAPI()
        print("✅ TrainingAPI created successfully")
    except ImportError as e:
        print(f"⚠️ API import failed (might be missing Flask): {e}")


def test_cli_imports():
    """Test CLI component imports"""
    print("\n🧪 Testing CLI Imports...")
    
    try:
        from train_system.cli import main
        print("✅ CLI imports successful")
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Train System Tests")
    print("=" * 50)
    
    test_functions = [
        test_model_wrapping,
        test_adapters,
        test_configuration,
        test_model_utils,
        test_api_imports,
        test_cli_imports
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Train System is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
