#!/usr/bin/env python3
"""
Setup script for Train System

A comprehensive training system that can train any model with any dataset
using configuration files and providing API access.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="train-system",
    version="1.0.0",
    author="AI Assistant",
    author_email="assistant@example.com",
    description="A comprehensive training system for PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/train-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
    ],
    python_requires=">=3.8",
    install_requires=requirements if requirements else [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "Pillow>=8.0.0",
        "PyYAML>=5.4.0",
        "tensorboard>=2.7.0",
        "tqdm>=4.60.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-system=train_system.cli.main:main",
            "ts-train=train_system.cli.main:train_command",
            "ts-api=train_system.api.server:run_server",
            "ts-template=train_system.cli.main:template_command",
            "ts-complete-config=train_system.cli.main:complete_config_command",
        ],
    },
    include_package_data=True,
    package_data={
        "train_system": [
            "config/templates/*.yaml",
            "config/templates/*.json",
        ],
    },
    zip_safe=False,
)
