"""
API module for Train System

Provides REST API functionality for remote training control.
Includes both basic API and enhanced web API for deployment.
"""

from .server import TrainingAPI, create_app, run_server
from .web_server import EnhancedTrainingAPI, create_web_app, run_web_server
from .config import get_config, APIConfig
from .deploy import deploy_api

__all__ = [
    # Basic API
    "TrainingAPI", 
    "create_app", 
    "run_server",
    
    # Enhanced Web API
    "EnhancedTrainingAPI",
    "create_web_app", 
    "run_web_server",
    
    # Configuration
    "get_config",
    "APIConfig",
    
    # Deployment
    "deploy_api"
]
