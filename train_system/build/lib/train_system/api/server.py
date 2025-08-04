"""
REST API Server for Train System

Provides REST API endpoints for controlling training via HTTP requests.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import queue
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
import time

from ..config import UnifiedTrainingConfig, ConfigTemplateManager
from ..core.trainer import UnifiedTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingAPI:
    """Training API server"""
    
    def __init__(self, host='localhost', port=5000):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.host = host
        self.port = port
        
        # Training state
        self.current_trainer = None
        self.training_thread = None
        self.training_status = "idle"  # idle, running, completed, error
        self.training_results = None
        self.training_error = None
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/', methods=['GET'])
        def index():
            """API documentation"""
            return jsonify({
                "name": "Train System API",
                "version": "1.0.0",
                "status": "running",
                "endpoints": {
                    "GET /": "API documentation",
                    "GET /status": "Get training status",
                    "POST /train": "Start training with config",
                    "POST /stop": "Stop current training",
                    "GET /results": "Get training results",
                    "GET /templates": "List available config templates",
                    "GET /templates/<name>": "Get specific config template",
                    "POST /validate": "Validate configuration"
                }
            })
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """Get current training status"""
            response = {
                "status": self.training_status,
                "timestamp": datetime.now().isoformat()
            }
            
            if self.current_trainer:
                response["training_info"] = {
                    "current_epoch": getattr(self.current_trainer, 'current_epoch', 0),
                    "total_epochs": self.current_trainer.config.training.epochs,
                    "best_val_acc": getattr(self.current_trainer, 'best_val_acc', 0.0),
                    "experiment_name": self.current_trainer.config.output.experiment_name
                }
            
            if self.training_error:
                response["error"] = self.training_error
            
            return jsonify(response)
        
        @self.app.route('/train', methods=['POST'])
        def start_training():
            """Start training with provided configuration"""
            if self.training_status == "running":
                return jsonify({
                    "error": "Training already in progress",
                    "status": self.training_status
                }), 400
            
            try:
                # Get configuration from request
                config_data = request.get_json()
                if not config_data:
                    return jsonify({"error": "No configuration provided"}), 400
                
                # Create configuration object
                config = UnifiedTrainingConfig.from_dict(config_data)
                
                # Validate configuration
                from ..config import ConfigValidator
                validation_result = ConfigValidator.validate(config)
                if not validation_result.is_valid:
                    return jsonify({
                        "error": "Invalid configuration",
                        "validation_errors": validation_result.errors,
                        "validation_warnings": validation_result.warnings
                    }), 400
                
                # Start training in background thread
                self.training_status = "running"
                self.training_error = None
                self.training_results = None
                
                self.current_trainer = UnifiedTrainer(config)
                self.training_thread = threading.Thread(
                    target=self._run_training,
                    args=(self.current_trainer,)
                )
                self.training_thread.start()
                
                return jsonify({
                    "message": "Training started successfully",
                    "status": self.training_status,
                    "experiment_name": config.output.experiment_name
                })
                
            except Exception as e:
                self.training_status = "error"
                self.training_error = str(e)
                logger.error(f"Error starting training: {e}")
                logger.error(traceback.format_exc())
                
                return jsonify({
                    "error": f"Failed to start training: {str(e)}",
                    "status": self.training_status
                }), 500
        
        @self.app.route('/stop', methods=['POST'])
        def stop_training():
            """Stop current training"""
            if self.training_status != "running":
                return jsonify({
                    "error": "No training in progress",
                    "status": self.training_status
                }), 400
            
            # Note: This is a simple implementation
            # In a production system, you'd want proper training interruption
            self.training_status = "stopped"
            
            return jsonify({
                "message": "Training stop requested",
                "status": self.training_status
            })
        
        @self.app.route('/results', methods=['GET'])
        def get_results():
            """Get training results"""
            if self.training_results is None:
                return jsonify({
                    "error": "No training results available",
                    "status": self.training_status
                }), 404
            
            return jsonify({
                "results": self.training_results,
                "status": self.training_status
            })
        
        @self.app.route('/templates', methods=['GET'])
        def list_templates():
            """List available configuration templates"""
            templates = ["blip", "generic", "torchvision"]
            return jsonify({
                "templates": templates,
                "descriptions": {
                    "blip": "BLIP model for deepfake detection",
                    "generic": "Generic model template",
                    "torchvision": "Torchvision model template"
                }
            })
        
        @self.app.route('/templates/<template_name>', methods=['GET'])
        def get_template(template_name):
            """Get specific configuration template"""
            try:
                template = ConfigTemplateManager.get_template(template_name)
                return jsonify(template)
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
        
        @self.app.route('/validate', methods=['POST'])
        def validate_config():
            """Validate configuration"""
            try:
                config_data = request.get_json()
                if not config_data:
                    return jsonify({"error": "No configuration provided"}), 400
                
                config = UnifiedTrainingConfig.from_dict(config_data)
                
                from ..config import ConfigValidator
                validation_result = ConfigValidator.validate(config)
                
                return jsonify({
                    "valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                })
                
            except Exception as e:
                return jsonify({
                    "error": f"Validation failed: {str(e)}",
                    "valid": False
                }), 400
        
        @self.app.route('/logs/<path:filename>', methods=['GET'])
        def get_log_file(filename):
            """Serve log files"""
            try:
                # Simple implementation - serve from training output directory
                if self.current_trainer:
                    log_dir = self.current_trainer.output_dir
                    return send_from_directory(log_dir, filename)
                else:
                    return jsonify({"error": "No active training session"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 404
    
    def _run_training(self, trainer: UnifiedTrainer):
        """Run training in background thread"""
        try:
            logger.info("Starting training in background thread")
            self.training_results = trainer.train()
            self.training_status = "completed"
            logger.info("Training completed successfully")
            
        except Exception as e:
            self.training_status = "error"
            self.training_error = str(e)
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())
    
    def run(self, debug=False):
        """Start the API server"""
        logger.info(f"Starting Train System API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


def create_app():
    """Create Flask app for external servers (gunicorn, etc.)"""
    api = TrainingAPI()
    return api.app


def run_server():
    """Entry point for console script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train System API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    api = TrainingAPI(host=args.host, port=args.port)
    api.run(debug=args.debug)


if __name__ == "__main__":
    run_server()
