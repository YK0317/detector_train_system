"""
Core components of the Train System

This module contains the core functionality for the training system including:
- UnifiedTrainingWrapper: Main wrapper for making any model trainable
- ModelFactory: Factory for creating wrapped models
- UnifiedTrainer: Training pipeline
- UnifiedDataset: Dataset handling
"""

from .dataset import UnifiedDataset
from .trainer import UnifiedTrainer
from .wrapper import ModelFactory, ModelUtils, UnifiedTrainingWrapper

__all__ = [
    "UnifiedTrainingWrapper",
    "ModelFactory",
    "ModelUtils",
    "UnifiedTrainer",
    "UnifiedDataset",
]
