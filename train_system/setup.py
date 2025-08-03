#!/usr/bin/env python3
"""
Setup script for Train System

A comprehensive training system that can train any model with any dataset
using configuration files and providing API access.
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core dependencies that are always required
core_requirements = [
    "torch>=1.9.0",
    "torchvision>=0.10.0", 
    "Pillow>=8.0.0",
    "PyYAML>=5.4.0",
    "tensorboard>=2.7.0",
    "tqdm>=4.60.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "psutil>=5.8.0",
    "pathlib-compat>=1.0.0;python_version<'3.4'",  # For older Python
]

# Web API dependencies
web_requirements = [
    "flask>=2.0.0",
    "flask-cors>=3.0.0",
    "requests>=2.25.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
    "black>=21.0.0",
    "isort>=5.0.0",
    "flake8>=3.8.0",
    "mypy>=0.800",
    "pre-commit>=2.15.0",
]

# Documentation dependencies
docs_requirements = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
    "myst-parser>=0.15.0",
]

# Optional model libraries
optional_requirements = [
    "timm>=0.6.0",  # For timm models
    "transformers>=4.0.0",  # For Hugging Face models
    "wandb>=0.12.0",  # For Weights & Biases logging
    "mlflow>=1.20.0",  # For MLflow tracking
    "opencv-python>=4.5.0",  # For advanced image processing
    "scikit-learn>=1.0.0",  # For metrics and utilities
    "matplotlib>=3.3.0",  # For plotting
    "seaborn>=0.11.0",  # For advanced plotting
]

# Use requirements from file if available, otherwise use core requirements
install_requires = requirements if requirements else core_requirements

setup(
    name="train-system",
    version="2.0.0",
    author="Train System Team", 
    author_email="contact@train-system.ai",
    description="A comprehensive, unified training system for PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/train-system/train-system",
    project_urls={
        "Bug Reports": "https://github.com/train-system/train-system/issues",
        "Source": "https://github.com/train-system/train-system",
        "Documentation": "https://train-system.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Framework :: Flask",
    ],
    keywords="pytorch training machine-learning deep-learning neural-networks",
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "web": web_requirements,
        "dev": dev_requirements,
        "docs": docs_requirements, 
        "optional": optional_requirements,
        "all": web_requirements + optional_requirements,
        "complete": web_requirements + dev_requirements + docs_requirements + optional_requirements,
    },
    entry_points={
        "console_scripts": [
            "train-system=train_system.cli.main:main",
            "ts-train=train_system.cli.main:train_command",
            "ts-dry-run=train_system.cli.main:dry_run_command",
            "ts-api=train_system.api.server:run_server",
            "ts-template=train_system.cli.main:template_command",
            "ts-config=train_system.cli.main:complete_config_command",
            "ts-validate=train_system.cli.main:validate_command",
            "ts-list=train_system.cli.main:list_command",
        ],
    },
    include_package_data=True,
    package_data={
        "train_system": [
            "config/templates/*.yaml",
            "config/templates/*.json", 
            "configs/*.yaml",
            "configs/*.json",
            "examples/*.py",
            "examples/*.yaml",
            "docs/*.md",
        ],
    },
    data_files=[
        ("share/train_system/configs", ["train_system/configs/default_config.yaml"]),
        ("share/train_system/examples", ["examples/basic_example.py"]),
    ] if Path("examples/basic_example.py").exists() else [],
    zip_safe=False,
    platforms=["any"],
    test_suite="tests",
)
