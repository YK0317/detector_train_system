"""
Test script for Enhanced Train System Web API

Validates the enhanced API functionality and endpoints.
"""

import json
import time
import requests
import threading
from pathlib import Path


class APITester:
    """Test the enhanced API functionality"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.session = requests.Session()
    
    def test_api_info(self):
        """Test API info endpoint"""
        print("Testing API info...")
        try:
            response = self.session.get(f"{self.api_base}/")
            if response.status_code == 200:
                info = response.json()
                print(f"✅ API Name: {info['name']}")
                print(f"✅ API Version: {info['version']}")
                return True
            else:
                print(f"❌ API info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API info error: {e}")
            return False
    
    def test_job_creation(self):
        """Test job creation"""
        print("\nTesting job creation...")
        
        test_config = {
            "model": {
                "name": "test_job",
                "type": "torchvision",
                "model_name": "resnet18",
                "num_classes": 2
            },
            "data": {
                "name": "test_data",
                "type": "image",
                "train_path": "/tmp/train",
                "val_path": "/tmp/val",
                "batch_size": 2
            },
            "training": {
                "epochs": 2,
                "learning_rate": 0.001
            },
            "output": {
                "output_dir": "/tmp/output",
                "experiment_name": "api_test"
            }
        }
        
        try:
            response = self.session.post(
                f"{self.api_base}/jobs",
                json=test_config
            )
            
            if response.status_code == 201:
                job_data = response.json()
                job_id = job_data['job_id']
                print(f"✅ Job created: {job_id}")
                return job_id
            else:
                print(f"❌ Job creation failed: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"❌ Job creation error: {e}")
            return None
    
    def test_job_listing(self):
        """Test job listing"""
        print("\nTesting job listing...")
        try:
            response = self.session.get(f"{self.api_base}/jobs")
            if response.status_code == 200:
                jobs = response.json()['jobs']
                print(f"✅ Found {len(jobs)} jobs")
                return jobs
            else:
                print(f"❌ Job listing failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Job listing error: {e}")
            return None
    
    def test_config_validation(self):
        """Test configuration validation"""
        print("\nTesting config validation...")
        
        # Test valid config
        valid_config = {
            "model": {"name": "test", "type": "torchvision", "model_name": "resnet18", "num_classes": 2},
            "data": {"name": "test", "type": "image", "batch_size": 32},
            "training": {"epochs": 10, "learning_rate": 0.001},
            "output": {"output_dir": "/tmp", "experiment_name": "test"}
        }
        
        try:
            response = self.session.post(
                f"{self.api_base}/config/validate",
                json=valid_config
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['valid']:
                    print("✅ Valid config correctly validated")
                else:
                    print(f"❌ Valid config rejected: {result['errors']}")
                return result['valid']
            else:
                print(f"❌ Config validation failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False
    
    def test_template_listing(self):
        """Test template listing"""
        print("\nTesting template listing...")
        try:
            response = self.session.get(f"{self.api_base}/config/templates")
            if response.status_code == 200:
                templates = response.json()
                print(f"✅ Found templates: {templates['templates']}")
                return True
            else:
                print(f"❌ Template listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Template listing error: {e}")
            return False
    
    def test_dataset_management(self):
        """Test dataset management (without actual file)"""
        print("\nTesting dataset listing...")
        try:
            response = self.session.get(f"{self.api_base}/datasets")
            if response.status_code == 200:
                datasets = response.json()['datasets']
                print(f"✅ Found {len(datasets)} datasets")
                return True
            else:
                print(f"❌ Dataset listing failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Dataset listing error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 50)
        print("Running Enhanced API Tests")
        print("=" * 50)
        
        tests = [
            ("API Info", self.test_api_info),
            ("Config Validation", self.test_config_validation),
            ("Template Listing", self.test_template_listing),
            ("Dataset Management", self.test_dataset_management),
            ("Job Creation", self.test_job_creation),
            ("Job Listing", self.test_job_listing),
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            try:
                result = test_func()
                results[test_name] = result is not False
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Results Summary")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
            if success:
                passed += 1
        
        print(f"\nPassed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️ {total - passed} tests failed")
        
        return passed == total


def start_test_server():
    """Start the API server for testing"""
    print("Starting test server...")
    try:
        from train_system.api.web_server import run_web_server
        import threading
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=run_web_server,
            kwargs={'host': 'localhost', 'port': 5000, 'debug': False}
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        print("✅ Test server started")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start test server: {e}")
        return False


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Enhanced Train System API")
    parser.add_argument('--url', default='http://localhost:5000', 
                       help='API base URL')
    parser.add_argument('--start-server', action='store_true',
                       help='Start test server automatically')
    
    args = parser.parse_args()
    
    if args.start_server:
        if not start_test_server():
            return False
    
    tester = APITester(args.url)
    return tester.run_all_tests()


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
