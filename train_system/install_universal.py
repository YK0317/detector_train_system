#!/usr/bin/env python3
"""
Universal Installation Script for Train-System
Supports: Local, Google Colab, Kaggle, Jupyter, and other environments
Run this in any environment: !python install_universal.py
"""

import subprocess
import sys
import os
import shutil
import platform
import fnmatch
import json
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class EnvironmentDetector:
    """Detect and analyze the current environment"""

    @staticmethod
    def detect_environment() -> str:
        """Detect the current environment type"""
        # Check for Kaggle
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            return "kaggle"

        # Check for Google Colab
        if (
            "COLAB_GPU" in os.environ
            or "COLAB_TPU_ADDR" in os.environ
            or os.path.exists("/content")
        ):
            return "colab"

        # Check for Jupyter
        if "JUPYTER_SERVER_ROOT" in os.environ or "JPY_PARENT_PID" in os.environ:
            return "jupyter"

        # Check for other cloud environments
        if "CLOUD_SHELL" in os.environ:
            return "cloud_shell"

        if "SAGEMAKER_NOTEBOOK_INSTANCE_NAME" in os.environ:
            return "sagemaker"

        return "local"

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get detailed platform information"""
        return {
            "system": platform.system(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "platform": platform.platform(),
            "node": platform.node(),
            "machine": platform.machine(),
        }

    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """Check GPU availability"""
        gpu_info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "total_memory": 0,
        }

        try:
            import torch

            gpu_info["cuda_available"] = torch.cuda.is_available()
            if gpu_info["cuda_available"]:
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["gpu_names"] = [
                    torch.cuda.get_device_name(i) for i in range(gpu_info["gpu_count"])
                ]
                gpu_info["total_memory"] = sum(
                    [
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(gpu_info["gpu_count"])
                    ]
                ) / (
                    1024**3
                )  # Convert to GB
        except ImportError:
            pass

        return gpu_info

    @staticmethod
    def get_resource_info() -> Dict[str, Any]:
        """Get system resource information"""
        resources = {"cpu_count": os.cpu_count(), "memory_info": {}}

        try:
            import psutil

            memory = psutil.virtual_memory()
            resources["memory_info"] = {
                "total": memory.total / (1024**3),  # GB
                "available": memory.available / (1024**3),  # GB
                "percent": memory.percent,
            }
        except ImportError:
            pass

        return resources


class CacheManager:
    """Comprehensive cache and build artifact manager"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.removed_items = []

    def clean_thoroughly(self) -> List[str]:
        """Perform comprehensive cleanup of all cache and build artifacts"""
        if self.verbose:
            print("Performing thorough cleanup...")

        # Define all patterns to clean
        build_artifacts = [
            "build",
            "dist",
            "*.egg-info",
            ".eggs",
            ".tox",
            "htmlcov",
            ".coverage*",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".black_cache",
        ]

        cache_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*~",
            ".DS_Store",
            "*.tmp",
            "*.temp",
            ".ipynb_checkpoints",
            "Thumbs.db",
            "*.swp",
            "*.swo",
            ".vscode/.ropeproject",
        ]

        temp_dirs = ["tmp", "temp", ".cache", ".pip"]

        all_patterns = build_artifacts + cache_patterns + temp_dirs

        # Clean current directory first
        self._clean_directory(".", all_patterns)

        # Clean recursively
        for root, dirs, files in os.walk("."):
            self._clean_items_in_directory(root, dirs, files, all_patterns)

        if self.verbose and self.removed_items:
            print(f"Cleaned {len(self.removed_items)} items")

        return self.removed_items

    def _clean_directory(self, directory: str, patterns: List[str]):
        """Clean items in a specific directory"""
        try:
            items = os.listdir(directory)
            for item in items:
                item_path = os.path.join(directory, item)
                for pattern in patterns:
                    if self._matches_pattern(item, pattern):
                        self._remove_item(item_path)
                        break
        except OSError:
            pass

    def _clean_items_in_directory(
        self, root: str, dirs: List[str], files: List[str], patterns: List[str]
    ):
        """Clean items in a directory during os.walk"""
        # Clean directories
        dirs_to_remove = []
        for dir_name in dirs:
            for pattern in patterns:
                if self._matches_pattern(dir_name, pattern) and "*" not in pattern:
                    dirs_to_remove.append(dir_name)
                    break

        for dir_name in dirs_to_remove:
            dir_path = os.path.join(root, dir_name)
            self._remove_item(dir_path)
            dirs.remove(dir_name)  # Don't recurse into removed directories

        # Clean files
        for file_name in files:
            for pattern in patterns:
                if self._matches_pattern(file_name, pattern):
                    file_path = os.path.join(root, file_name)
                    self._remove_item(file_path)
                    break

    def _matches_pattern(self, item: str, pattern: str) -> bool:
        """Check if item matches pattern"""
        if "*" in pattern:
            return fnmatch.fnmatch(item, pattern)
        else:
            return item == pattern

    def _remove_item(self, path: str):
        """Safely remove a file or directory"""
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

            self.removed_items.append(path)
            if self.verbose:
                print(f"Removed: {path}")
        except OSError as e:
            if self.verbose:
                print(f"Failed to remove {path}: {e}")


class DependencyResolver:
    """Resolve and install dependencies based on environment"""

    def __init__(self, environment: str, platform_info: Dict[str, Any]):
        self.environment = environment
        self.platform_info = platform_info
        self.base_requirements = self._get_base_requirements()
        self.env_requirements = self._get_environment_requirements()

    def _get_base_requirements(self) -> List[str]:
        """Get base requirements for all environments"""
        return [
            "torch>=1.9.0,<2.5.0",
            "torchvision>=0.10.0,<0.20.0",
            "numpy>=1.21.0,<1.27.0",
            "Pillow>=8.3.0,<11.0.0",
            "PyYAML>=6.0,<7.0",
            "tqdm>=4.62.0,<5.0.0",
            "tensorboard>=2.7.0,<3.0.0",
            "setuptools>=61.0",
            "wheel>=0.37.0",
            "packaging>=21.0",
        ]

    def _get_environment_requirements(self) -> List[str]:
        """Get environment-specific requirements"""
        env_specific = {
            "kaggle": [
                "matplotlib>=3.4.0,<4.0.0",
                "seaborn>=0.11.0,<1.0.0",
                "scikit-learn>=1.0.0,<2.0.0",
                "pandas>=1.3.0,<2.3.0",
                "opencv-python>=4.5.0,<5.0.0",
            ],
            "colab": [
                "matplotlib>=3.4.0,<4.0.0",
                "ipywidgets>=7.6.0,<9.0.0",
                "opencv-python>=4.5.0,<5.0.0",
            ],
            "local": [
                "matplotlib>=3.4.0,<4.0.0",
                "seaborn>=0.11.0,<1.0.0",
                "flask>=2.0.0,<4.0.0",
                "werkzeug>=1.0.1,<4.0.0",
                "psutil>=5.8.0,<6.0.0",
            ],
            "jupyter": [
                "matplotlib>=3.4.0,<4.0.0",
                "ipywidgets>=7.6.0,<9.0.0",
                "jupyter>=1.0.0",
            ],
        }

        return env_specific.get(self.environment, env_specific["local"])

    def get_all_requirements(self) -> List[str]:
        """Get all requirements for current environment"""
        return self.base_requirements + self.env_requirements

    def install_dependencies(self) -> bool:
        """Install all dependencies with proper error handling"""
        print(f"Installing dependencies for {self.environment} environment...")

        try:
            # Upgrade core tools first
            core_tools = ["pip", "setuptools", "wheel"]
            print("Upgrading core tools...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade"]
                + core_tools
                + ["--quiet"]
            )

            # Install requirements in batches to handle conflicts better
            all_requirements = self.get_all_requirements()

            for requirement in all_requirements:
                try:
                    print(f"Installing {requirement}...")
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            requirement,
                            "--quiet",
                            "--no-warn-script-location",
                        ]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"WARNING: Failed to install {requirement}: {e}")
                    # Continue with other packages

            print("Dependencies installed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install core dependencies: {e}")
            return False


class UniversalInstaller:
    """Universal installer that adapts to different environments"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.detector = EnvironmentDetector()
        self.cache_manager = CacheManager(verbose)

        # Detect environment
        self.environment = self.detector.detect_environment()
        self.platform_info = self.detector.get_platform_info()
        self.gpu_info = self.detector.check_gpu_availability()
        self.resource_info = self.detector.get_resource_info()

        # Initialize dependency resolver
        self.dependency_resolver = DependencyResolver(
            self.environment, self.platform_info
        )

    def print_environment_info(self):
        """Print detailed environment information"""
        print(f"Environment Information")
        print("=" * 50)
        print(f"Environment Type: {self.environment}")
        print(
            f"Platform: {self.platform_info['system']} {self.platform_info['architecture']}"
        )
        print(f"Python: {self.platform_info['python_version'].split()[0]}")
        print(f"Node: {self.platform_info['node']}")

        if self.gpu_info["cuda_available"]:
            print(f"GPU: {', '.join(self.gpu_info['gpu_names'])}")
            print(f"CUDA: {self.gpu_info['cuda_version']}")
            print(f"GPU Memory: {self.gpu_info['total_memory']:.1f} GB")
        else:
            print("GPU: Not available")

        if "total" in self.resource_info["memory_info"]:
            memory_gb = self.resource_info["memory_info"]["total"]
            print(f"RAM: {memory_gb:.1f} GB")

        print(f"CPU Cores: {self.resource_info['cpu_count']}")
        print()

    def clean_environment(self) -> bool:
        """Clean the environment thoroughly"""
        try:
            removed_items = self.cache_manager.clean_thoroughly()
            return True
        except Exception as e:
            print(f"⚠️  Cleanup encountered issues: {e}")
            return False

    def install_train_system(self) -> bool:
        """Install train-system package"""
        print("Installing train-system package...")

        try:
            # Install in development mode with proper flags
            install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "--no-build-isolation",
                "--no-deps",
            ]

            if self.verbose:
                install_cmd.append("--verbose")
            else:
                install_cmd.append("--quiet")

            subprocess.check_call(install_cmd)
            print("Train-system package installed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Package installation failed: {e}")
            return False

    def verify_installation(self) -> bool:
        """Verify the installation works correctly"""
        print("Verifying installation...")

        try:
            # Test import
            import train_system

            version = getattr(train_system, "__version__", "Unknown")
            print(f"Import successful - Version: {version}")

            # Test CLI
            result = subprocess.run(
                [sys.executable, "-m", "train_system.cli.main", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print("CLI test passed")
            else:
                print(f"WARNING: CLI test failed: {result.stderr}")
                return False

            return True

        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            return False

    def create_environment_config(self):
        """Create environment-specific configuration"""
        print("Creating environment configuration...")

        config_templates = {
            "kaggle": {
                "environment": "kaggle",
                "data_dir": "/kaggle/input",
                "output_dir": "/kaggle/working",
                "num_workers": 2,
                "pin_memory": False,
                "batch_size": 32,
                "persistent_workers": False,
            },
            "colab": {
                "environment": "colab",
                "data_dir": "/content/data",
                "output_dir": "/content/outputs",
                "num_workers": 2,
                "pin_memory": True,
                "batch_size": 32,
                "persistent_workers": False,
            },
            "local": {
                "environment": "local",
                "data_dir": "./data",
                "output_dir": "./outputs",
                "num_workers": min(4, self.resource_info["cpu_count"]),
                "pin_memory": True,
                "batch_size": 64,
                "persistent_workers": True,
            },
        }

        config = config_templates.get(self.environment, config_templates["local"])

        # Add GPU information
        config["gpu_available"] = self.gpu_info["cuda_available"]
        config["gpu_count"] = self.gpu_info["gpu_count"]

        # Save environment config
        try:
            import yaml

            config_path = f"{self.environment}_defaults.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Environment config saved: {config_path}")
        except Exception as e:
            print(f"WARNING: Failed to save config: {e}")

    def install(self) -> bool:
        """Main installation method"""
        print("Universal Train-System Installation")
        print("=" * 60)

        # Print environment info
        self.print_environment_info()

        # Clean environment
        if not self.clean_environment():
            print("WARNING: Environment cleanup had issues, continuing...")

        # Install dependencies
        if not self.dependency_resolver.install_dependencies():
            print("ERROR: Failed to install dependencies")
            return False

        # Install train-system package
        if not self.install_train_system():
            print("ERROR: Failed to install train-system package")
            return False

        # Verify installation
        if not self.verify_installation():
            print("ERROR: Installation verification failed")
            return False

        # Create environment config
        self.create_environment_config()

        print("\nInstallation completed successfully!")
        self._print_next_steps()

        return True

    def _print_next_steps(self):
        """Print environment-specific next steps"""
        next_steps = {
            "kaggle": [
                "1. Upload your dataset to Kaggle Datasets",
                "2. Add the dataset to your notebook",
                "3. Edit kaggle_defaults.yaml with your dataset paths",
                "4. Run: !python -m train_system.cli.main dry-run your_config.yaml",
                "5. Run: !python -m train_system.cli.main train your_config.yaml",
            ],
            "colab": [
                "1. Mount Google Drive: from google.colab import drive; drive.mount('/content/drive')",
                "2. Upload your data to /content/data or use drive",
                "3. Create your training config file",
                "4. Run: !python -m train_system.cli.main dry-run your_config.yaml",
                "5. Run: !python -m train_system.cli.main train your_config.yaml",
            ],
            "local": [
                "1. Prepare your dataset in ./data directory",
                "2. Create your training configuration file",
                "3. Test with: python -m train_system.cli.main dry-run your_config.yaml",
                "4. Start training: python -m train_system.cli.main train your_config.yaml",
            ],
        }

        print(f"\nNext steps for {self.environment}:")
        steps = next_steps.get(self.environment, next_steps["local"])
        for step in steps:
            print(f"   {step}")

        print(
            f"\nTIP: Environment-specific config created: {self.environment}_defaults.yaml"
        )
        print("Documentation: https://train-system.readthedocs.io/")


def main():
    """Main entry point"""
    installer = UniversalInstaller(verbose=True)
    success = installer.install()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
