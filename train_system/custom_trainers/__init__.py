"""
Custom trainers for the train_system package.

This module contains external training implementations that can be used
via the external trainer system.
"""

from .yolo_trainer import YOLOBuiltinTrainer, CustomYOLOTrainer

__all__ = ["YOLOBuiltinTrainer", "CustomYOLOTrainer"]
