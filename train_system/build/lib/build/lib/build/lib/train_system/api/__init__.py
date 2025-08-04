"""
API module for Train System

Provides REST API functionality for remote training control.
"""

from .server import TrainingAPI, create_app, run_server

__all__ = ["TrainingAPI", "create_app", "run_server"]
