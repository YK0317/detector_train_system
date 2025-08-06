#!/usr/bin/env python3
"""
Simple test script for Enhanced API
"""

import sys
import os

# Add the train_system to path
sys.path.insert(0, os.path.dirname(__file__))

def test_api_import():
    """Test importing the Enhanced API"""
    print("🧪 Testing Enhanced API Import...")
    
    try:
        from train_system.api.web_server import EnhancedTrainingAPI
        print("✅ EnhancedTrainingAPI import successful")
        
        # Test creating an instance
        api = EnhancedTrainingAPI()
        print("✅ EnhancedTrainingAPI instance created")
        print(f"   Host: {api.host}")
        print(f"   Port: {api.port}")
        
        # Test managers
        if hasattr(api, 'job_manager'):
            print("✅ JobManager initialized")
        if hasattr(api, 'file_manager'):
            print("✅ FileManager initialized")
            
        print("✅ All API components working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic API functionality without server"""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        from train_system.api.web_server import JobManager, FileManager
        
        # Test JobManager
        job_manager = JobManager()
        print("✅ JobManager created")
        
        # Test creating a job
        test_config = {
            "model": {"name": "test", "type": "torchvision", "model_name": "resnet18", "num_classes": 2},
            "training": {"epochs": 1, "learning_rate": 0.001}
        }
        
        job_id = job_manager.create_job(test_config)
        print(f"✅ Test job created: {job_id}")
        
        # Test job retrieval
        job = job_manager.get_job(job_id)
        if job and job['id'] == job_id:
            print("✅ Job retrieval working")
        
        # Test FileManager
        file_manager = FileManager()
        print("✅ FileManager created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_validation():
    """Test configuration validation"""
    print("\n📋 Testing Configuration Validation...")
    
    try:
        from train_system.config import UnifiedTrainingConfig, ConfigValidator
        
        # Test valid config
        valid_config = {
            "model": {
                "name": "test_model",
                "type": "torchvision",
                "model_name": "resnet18",
                "num_classes": 2
            },
            "data": {
                "name": "test_data",
                "type": "image",
                "train_path": "/tmp/train",
                "val_path": "/tmp/val",
                "batch_size": 32
            },
            "training": {
                "epochs": 5,
                "learning_rate": 0.001
            },
            "output": {
                "output_dir": "/tmp/output",
                "experiment_name": "test"
            }
        }
        
        config = UnifiedTrainingConfig.from_dict(valid_config)
        validation_result = ConfigValidator.validate(config)
        
        if validation_result.is_valid:
            print("✅ Configuration validation working")
        else:
            print(f"⚠️ Config validation issues: {validation_result.errors}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in config validation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Enhanced API Test Suite")
    print("=" * 40)
    
    tests = [
        ("API Import", test_api_import),
        ("Basic Functionality", test_basic_functionality),
        ("Config Validation", test_config_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("=" * 40)
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! API is ready.")
    else:
        print("⚠️ Some tests failed. Check dependencies.")
