#!/usr/bin/env python3
"""
Installation script for Google Colab environment
Run this in a Colab cell: !python install_colab.py
"""

import subprocess
import sys
import os
import shutil

def clean_build_artifacts():
    """Clean up any existing build artifacts and cache files"""
    artifacts = [
        'build',
        'dist', 
        'train_system.egg-info',
        '*.egg-info',
        '.eggs'
    ]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
                print(f"Removed directory: {artifact}")
            else:
                os.remove(artifact)
                print(f"Removed file: {artifact}")
    
    # Clean pycache directories more thoroughly
    for root, dirs, files in os.walk('.'):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path)
            print(f"Removed: {pycache_path}")
        
        # Remove .pyc files
        for file in files:
            if file.endswith(('.pyc', '.pyo', '.pyd')):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")
    
    # Remove other cache and temporary files
    temp_patterns = [
        '.coverage',
        '.pytest_cache',
        '.mypy_cache',
        '.tox',
        'htmlcov',
        '*.tmp',
        '*.temp',
        '*~',
        '.DS_Store'
    ]
    
    for root, dirs, files in os.walk('.'):
        for pattern in temp_patterns:
            if '*' in pattern:
                # Handle wildcard patterns
                import fnmatch
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f"Removed: {file_path}")
            else:
                # Handle directory patterns
                if pattern in dirs:
                    dir_path = os.path.join(root, pattern)
                    shutil.rmtree(dir_path)
                    print(f"Removed: {dir_path}")
                # Handle file patterns
                if pattern in files:
                    file_path = os.path.join(root, pattern)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")

def install_package():
    """Install the package in development mode"""
    try:
        # Clean first
        clean_build_artifacts()
        
        # Upgrade pip and setuptools
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
        
        # Install in editable mode with verbose output
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.', '--verbose'])
        
        print("✅ Train-system installed successfully!")
        
        # Test the installation
        try:
            import train_system
            print(f"✅ Import test passed. Version: {train_system.__version__ if hasattr(train_system, '__version__') else 'Unknown'}")
        except ImportError as e:
            print(f"❌ Import test failed: {e}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Installing train-system for Google Colab...")
    success = install_package()
    
    if success:
        print("\n📋 Next steps:")
        print("1. Import train_system in your notebook")
        print("2. Create your training config (see examples/)")
        print("3. Run: python -m train_system.cli.main dry-run your_config.yaml")
        print("4. Run: python -m train_system.cli.main train your_config.yaml")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")
