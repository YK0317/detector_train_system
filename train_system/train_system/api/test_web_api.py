"""
Test script for Enhanced Web API

Tests all endpoints and functionality of the enhanced training API.
"""

import json
import requests
import time
import io
import zipfile
from pathlib import Path
import threading
from datetime import datetime

class APITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.test_results = []
        
    def log_test(self, test_name, success, message="", response_data=None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if response_data and not success:
            print(f"   Response: {response_data}")
    
    def test_api_info(self):
        """Test API info endpoint"""
        try:
            response = requests.get(f"{self.api_url}/")
            if response.status_code == 200:
                data = response.json()
                self.log_test("API Info", True, f"API version: {data.get('version', 'unknown')}", data)
                return True
            else:
                self.log_test("API Info", False, f"Status: {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("API Info", False, f"Connection error: {str(e)}")
            return False
    
    def test_config_validation(self):
        """Test configuration validation"""
        try:
            # Valid config
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
                    "experiment_name": "test_experiment"
                }
            }
            
            response = requests.post(f"{self.api_url}/config/validate", json=valid_config)
            if response.status_code == 200:
                data = response.json()
                if data.get('valid'):
                    self.log_test("Config Validation (Valid)", True, "Configuration validated successfully")
                else:
                    self.log_test("Config Validation (Valid)", False, "Valid config rejected", data)
            else:
                self.log_test("Config Validation (Valid)", False, f"Status: {response.status_code}")
            
            # Invalid config
            invalid_config = {
                "model": {"name": "test"},  # Missing required fields
                "data": {},
                "training": {}
            }
            
            response = requests.post(f"{self.api_url}/config/validate", json=invalid_config)
            if response.status_code == 400:
                data = response.json()
                if not data.get('valid'):
                    self.log_test("Config Validation (Invalid)", True, "Invalid config properly rejected")
                else:
                    self.log_test("Config Validation (Invalid)", False, "Invalid config accepted", data)
            else:
                self.log_test("Config Validation (Invalid)", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Config Validation", False, f"Error: {str(e)}")
    
    def test_config_templates(self):
        """Test configuration templates"""
        try:
            response = requests.get(f"{self.api_url}/config/templates")
            if response.status_code == 200:
                data = response.json()
                templates = data.get('templates', [])
                if isinstance(templates, list) and len(templates) > 0:
                    self.log_test("Config Templates", True, f"Found {len(templates)} templates")
                else:
                    self.log_test("Config Templates", False, "No templates found", data)
            else:
                self.log_test("Config Templates", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Config Templates", False, f"Error: {str(e)}")
    
    def test_job_creation(self):
        """Test job creation"""
        try:
            config = {
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
                    "batch_size": 2
                },
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.001
                },
                "output": {
                    "output_dir": "/tmp/output",
                    "experiment_name": "test_job"
                }
            }
            
            response = requests.post(f"{self.api_url}/jobs", json=config)
            if response.status_code == 201:
                data = response.json()
                job_id = data.get('job_id')
                if job_id:
                    self.log_test("Job Creation", True, f"Job created with ID: {job_id}")
                    return job_id
                else:
                    self.log_test("Job Creation", False, "No job ID returned", data)
            else:
                self.log_test("Job Creation", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Job Creation", False, f"Error: {str(e)}")
        return None
    
    def test_job_listing(self):
        """Test job listing"""
        try:
            response = requests.get(f"{self.api_url}/jobs")
            if response.status_code == 200:
                data = response.json()
                jobs = data.get('jobs', [])
                self.log_test("Job Listing", True, f"Found {len(jobs)} jobs")
                return jobs
            else:
                self.log_test("Job Listing", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Job Listing", False, f"Error: {str(e)}")
        return []
    
    def test_job_details(self, job_id):
        """Test getting job details"""
        if not job_id:
            self.log_test("Job Details", False, "No job ID provided")
            return
            
        try:
            response = requests.get(f"{self.api_url}/jobs/{job_id}")
            if response.status_code == 200:
                data = response.json()
                job = data.get('job')
                if job and job.get('id') == job_id:
                    self.log_test("Job Details", True, f"Job details retrieved for {job_id}")
                else:
                    self.log_test("Job Details", False, "Invalid job data", data)
            else:
                self.log_test("Job Details", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Job Details", False, f"Error: {str(e)}")
    
    def test_dataset_listing(self):
        """Test dataset listing"""
        try:
            response = requests.get(f"{self.api_url}/datasets")
            if response.status_code == 200:
                data = response.json()
                datasets = data.get('datasets', [])
                self.log_test("Dataset Listing", True, f"Found {len(datasets)} datasets")
                return datasets
            else:
                self.log_test("Dataset Listing", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Dataset Listing", False, f"Error: {str(e)}")
        return []
    
    def test_dataset_upload(self):
        """Test dataset upload with dummy file"""
        try:
            # Create a dummy zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                zip_file.writestr('train/class1/dummy.txt', 'dummy training data')
                zip_file.writestr('val/class1/dummy.txt', 'dummy validation data')
                zip_file.writestr('train/class2/dummy.txt', 'dummy training data 2')
                zip_file.writestr('val/class2/dummy.txt', 'dummy validation data 2')
            
            zip_buffer.seek(0)
            
            files = {'file': ('test_dataset.zip', zip_buffer, 'application/zip')}
            data = {'metadata': json.dumps({'description': 'Test dataset', 'classes': 2})}
            
            response = requests.post(f"{self.api_url}/datasets", files=files, data=data)
            if response.status_code == 201:
                result = response.json()
                dataset_id = result.get('dataset_id')
                if dataset_id:
                    self.log_test("Dataset Upload", True, f"Dataset uploaded with ID: {dataset_id}")
                    return dataset_id
                else:
                    self.log_test("Dataset Upload", False, "No dataset ID returned", result)
            else:
                self.log_test("Dataset Upload", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Dataset Upload", False, f"Error: {str(e)}")
        return None
    
    def test_api_imports(self):
        """Test API module imports"""
        try:
            from train_system.api.web_server import EnhancedTrainingAPI
            api = EnhancedTrainingAPI()
            self.log_test("API Imports", True, "EnhancedTrainingAPI imported successfully")
        except ImportError as e:
            self.log_test("API Imports", False, f"Import failed: {str(e)}")
        except Exception as e:
            self.log_test("API Imports", False, f"Error: {str(e)}")
    
    def run_all_tests(self, run_server_tests=True):
        """Run all tests"""
        print("🧪 Starting Enhanced API Tests")
        print("=" * 50)
        
        # Test imports first
        self.test_api_imports()
        
        if not run_server_tests:
            print("\n📊 Test Summary (Import Tests Only)")
            self.print_summary()
            return
        
        # Test if server is running
        if not self.test_api_info():
            print("❌ Server not running. Please start the API server first:")
            print("   python -m train_system.api.web_server")
            return
        
        # Run API tests
        print("\n🔧 Testing API endpoints...")
        self.test_config_templates()
        self.test_config_validation()
        
        print("\n📁 Testing dataset management...")
        self.test_dataset_listing()
        dataset_id = self.test_dataset_upload()
        if dataset_id:
            self.test_dataset_listing()  # Test again to see uploaded dataset
        
        print("\n🏃 Testing job management...")
        job_id = self.test_job_creation()
        self.test_job_listing()
        if job_id:
            self.test_job_details(job_id)
        
        print("\n📊 Test Summary")
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   - {result['test']}: {result['message']}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n📈 Success Rate: {success_rate:.1f}%")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Enhanced Web API")
    parser.add_argument("--url", default="http://localhost:5000", help="API base URL")
    parser.add_argument("--imports-only", action="store_true", help="Test imports only")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    tester.run_all_tests(run_server_tests=not args.imports_only)


if __name__ == "__main__":
    main()
