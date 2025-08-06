"""
Enhanced Web API Server for Train System

Provides comprehensive REST API endpoints for web-based training interface.
Includes file upload, job management, real-time updates, and model management.
"""

import json
import logging
import os
import uuid
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

try:
    from werkzeug.utils import secure_filename
except ImportError:
    print("Warning: werkzeug not available, using basic filename sanitization")
    def secure_filename(filename):
        return filename.replace('/', '_').replace('\\', '_')

import yaml

try:
    from flask import Flask, jsonify, request, send_from_directory, send_file
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    print("Warning: Flask components not available")
    FLASK_AVAILABLE = False

try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    SOCKETIO_AVAILABLE = True
except ImportError:
    print("Warning: flask-socketio not available, real-time features disabled")
    SOCKETIO_AVAILABLE = False

from ..config import ConfigTemplateManager, UnifiedTrainingConfig, ConfigValidator
from ..core.trainer import UnifiedTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobManager:
    """Manages multiple training jobs"""
    
    def __init__(self, storage_dir: str = "training_jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.jobs = {}
        self.job_threads = {}
        
    def create_job(self, config_data: dict, dataset_id: Optional[str] = None) -> str:
        """Create a new training job"""
        job_id = str(uuid.uuid4())
        
        job_info = {
            'id': job_id,
            'status': 'queued',
            'config': config_data,
            'dataset_id': dataset_id,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'current_epoch': 0,
            'total_epochs': config_data.get('training', {}).get('epochs', 0),
            'current_loss': None,
            'best_accuracy': 0.0,
            'error': None,
            'results': None,
            'progress_history': [],
            'logs': []
        }
        
        self.jobs[job_id] = job_info
        return job_id
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job information"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[dict]:
        """List all jobs"""
        return list(self.jobs.values())
    
    def update_job_progress(self, job_id: str, progress: dict):
        """Update job progress"""
        if job_id in self.jobs:
            self.jobs[job_id].update(progress)
            self.jobs[job_id]['progress_history'].append({
                'timestamp': datetime.now().isoformat(),
                **progress
            })
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        if job_id in self.jobs:
            # Stop job if running
            if job_id in self.job_threads:
                # In production, implement proper thread stopping
                pass
            
            # Clean up files
            job_dir = self.storage_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
            
            del self.jobs[job_id]
            return True
        return False


class FileManager:
    """Manages file uploads and storage"""
    
    def __init__(self, storage_dir: str = "uploaded_files"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.datasets_dir = self.storage_dir / "datasets"
        self.models_dir = self.storage_dir / "models"
        self.datasets_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.allowed_extensions = {
            'datasets': {'zip', 'tar', 'tar.gz'},
            'models': {'pth', 'pt', 'pkl', 'onnx'},
            'configs': {'yaml', 'yml', 'json'}
        }
    
    def allowed_file(self, filename: str, file_type: str) -> bool:
        """Check if file extension is allowed"""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.allowed_extensions.get(file_type, set()))
    
    def store_dataset(self, file, metadata: dict = None) -> str:
        """Store uploaded dataset"""
        if not self.allowed_file(file.filename, 'datasets'):
            raise ValueError(f"Invalid file type: {file.filename}")
        
        dataset_id = str(uuid.uuid4())
        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        filename = secure_filename(file.filename)
        file_path = dataset_dir / filename
        file.save(str(file_path))
        
        # Store metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'id': dataset_id,
                'filename': filename,
                'uploaded_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }, f)
        
        return dataset_id
    
    def get_dataset_info(self, dataset_id: str) -> Optional[dict]:
        """Get dataset information"""
        metadata_file = self.datasets_dir / dataset_id / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_datasets(self) -> List[dict]:
        """List all stored datasets"""
        datasets = []
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                info = self.get_dataset_info(dataset_dir.name)
                if info:
                    datasets.append(info)
        return datasets
    
    def store_model(self, file, metadata: dict = None) -> str:
        """Store uploaded model"""
        if not self.allowed_file(file.filename, 'models'):
            raise ValueError(f"Invalid file type: {file.filename}")
        
        model_id = str(uuid.uuid4())
        model_dir = self.models_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        filename = secure_filename(file.filename)
        file_path = model_dir / filename
        file.save(str(file_path))
        
        # Store metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'id': model_id,
                'filename': filename,
                'uploaded_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }, f)
        
        return model_id
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get model file path"""
        info = self.get_model_info(model_id)
        if info:
            return self.models_dir / model_id / info['filename']
        return None
    
    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get model information"""
        metadata_file = self.models_dir / model_id / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None


class EnhancedTrainingAPI:
    """Enhanced Training API server for web deployment"""

    def __init__(self, host="localhost", port=5000, debug=False):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the Enhanced API. Install with: pip install flask flask-cors")
        
        self.app = Flask(__name__)
        CORS(self.app, origins="*")
        
        if SOCKETIO_AVAILABLE:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        else:
            self.socketio = None
            logger.warning("SocketIO not available - real-time features disabled")

        self.host = host
        self.port = port
        self.debug = debug

        # Managers
        self.job_manager = JobManager()
        self.file_manager = FileManager()

        # Setup routes and socket handlers
        self._setup_routes()
        if self.socketio:
            self._setup_socket_handlers()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.route("/api/v1/", methods=["GET"])
        def index():
            """API documentation"""
            return jsonify({
                "name": "Train System Web API",
                "version": "2.0.0",
                "status": "running",
                "features": ["file_upload", "job_management", "real_time_updates"],
                "endpoints": {
                    # Job Management
                    "POST /api/v1/jobs": "Create new training job",
                    "GET /api/v1/jobs": "List all jobs",
                    "GET /api/v1/jobs/<id>": "Get job details",
                    "DELETE /api/v1/jobs/<id>": "Delete job",
                    "POST /api/v1/jobs/<id>/start": "Start job",
                    "POST /api/v1/jobs/<id>/stop": "Stop job",
                    
                    # File Management
                    "POST /api/v1/datasets": "Upload dataset",
                    "GET /api/v1/datasets": "List datasets",
                    "GET /api/v1/datasets/<id>": "Get dataset info",
                    "DELETE /api/v1/datasets/<id>": "Delete dataset",
                    
                    # Model Management
                    "POST /api/v1/models": "Upload model",
                    "GET /api/v1/models": "List models",
                    "GET /api/v1/models/<id>": "Get model info",
                    "GET /api/v1/models/<id>/download": "Download model",
                    
                    # Configuration
                    "GET /api/v1/config/templates": "List config templates",
                    "POST /api/v1/config/validate": "Validate configuration",
                    "POST /api/v1/config/generate": "Generate configuration"
                }
            })

        # Job Management Endpoints
        @self.app.route("/api/v1/jobs", methods=["POST"])
        def create_job():
            """Create new training job"""
            try:
                # Handle both JSON and form data
                if request.is_json:
                    config_data = request.get_json()
                    dataset_id = config_data.get('dataset_id')
                else:
                    config_data = json.loads(request.form.get('config', '{}'))
                    dataset_id = request.form.get('dataset_id')
                    
                    # Handle file upload
                    if 'dataset' in request.files:
                        file = request.files['dataset']
                        if file.filename:
                            dataset_id = self.file_manager.store_dataset(file)

                if not config_data:
                    return jsonify({"error": "No configuration provided"}), 400

                # Validate configuration
                config = UnifiedTrainingConfig.from_dict(config_data)
                validation_result = ConfigValidator.validate(config)
                
                if not validation_result.is_valid:
                    return jsonify({
                        "error": "Invalid configuration",
                        "validation_errors": validation_result.errors,
                        "validation_warnings": validation_result.warnings,
                    }), 400

                # Create job
                job_id = self.job_manager.create_job(config_data, dataset_id)

                return jsonify({
                    "job_id": job_id,
                    "status": "created",
                    "dataset_id": dataset_id,
                    "message": "Job created successfully"
                }), 201

            except Exception as e:
                logger.error(f"Error creating job: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/jobs", methods=["GET"])
        def list_jobs():
            """List all jobs"""
            try:
                jobs = self.job_manager.list_jobs()
                return jsonify({"jobs": jobs})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/jobs/<job_id>", methods=["GET"])
        def get_job(job_id):
            """Get job details"""
            try:
                job = self.job_manager.get_job(job_id)
                if not job:
                    return jsonify({"error": "Job not found"}), 404
                return jsonify({"job": job})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/jobs/<job_id>/start", methods=["POST"])
        def start_job(job_id):
            """Start training job"""
            try:
                job = self.job_manager.get_job(job_id)
                if not job:
                    return jsonify({"error": "Job not found"}), 404

                if job['status'] == 'running':
                    return jsonify({"error": "Job already running"}), 400

                # Start training in background thread
                thread = threading.Thread(target=self._run_training_job, args=(job_id,))
                thread.start()
                self.job_manager.job_threads[job_id] = thread

                # Update job status
                self.job_manager.update_job_progress(job_id, {
                    'status': 'running',
                    'started_at': datetime.now().isoformat()
                })

                return jsonify({"message": "Job started", "job_id": job_id})

            except Exception as e:
                logger.error(f"Error starting job {job_id}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/jobs/<job_id>", methods=["DELETE"])
        def delete_job(job_id):
            """Delete job"""
            try:
                success = self.job_manager.delete_job(job_id)
                if success:
                    return jsonify({"message": "Job deleted"})
                else:
                    return jsonify({"error": "Job not found"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # Dataset Management Endpoints
        @self.app.route("/api/v1/datasets", methods=["POST"])
        def upload_dataset():
            """Upload dataset"""
            try:
                if 'file' not in request.files:
                    return jsonify({"error": "No file provided"}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400

                metadata = json.loads(request.form.get('metadata', '{}'))
                dataset_id = self.file_manager.store_dataset(file, metadata)

                return jsonify({
                    "dataset_id": dataset_id,
                    "message": "Dataset uploaded successfully",
                    "info": self.file_manager.get_dataset_info(dataset_id)
                }), 201

            except Exception as e:
                logger.error(f"Error uploading dataset: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/datasets", methods=["GET"])
        def list_datasets():
            """List all datasets"""
            try:
                datasets = self.file_manager.list_datasets()
                return jsonify({"datasets": datasets})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/datasets/<dataset_id>", methods=["GET"])
        def get_dataset_info(dataset_id):
            """Get dataset information"""
            try:
                info = self.file_manager.get_dataset_info(dataset_id)
                if not info:
                    return jsonify({"error": "Dataset not found"}), 404
                return jsonify({"dataset": info})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # Model Management Endpoints
        @self.app.route("/api/v1/models/<model_id>/download", methods=["GET"])
        def download_model(model_id):
            """Download trained model"""
            try:
                model_path = self.file_manager.get_model_path(model_id)
                if not model_path or not model_path.exists():
                    return jsonify({"error": "Model not found"}), 404

                return send_file(str(model_path), as_attachment=True)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # Configuration Endpoints
        @self.app.route("/api/v1/config/templates", methods=["GET"])
        def list_config_templates():
            """List available configuration templates"""
            try:
                templates = ["blip", "generic", "torchvision"]
                return jsonify({
                    "templates": templates,
                    "descriptions": {
                        "blip": "BLIP model for deepfake detection",
                        "generic": "Generic model template",
                        "torchvision": "Torchvision model template",
                    },
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/v1/config/validate", methods=["POST"])
        def validate_config():
            """Validate configuration"""
            try:
                config_data = request.get_json()
                if not config_data:
                    return jsonify({"error": "No configuration provided"}), 400

                config = UnifiedTrainingConfig.from_dict(config_data)
                validation_result = ConfigValidator.validate(config)

                return jsonify({
                    "valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings,
                })

            except Exception as e:
                return jsonify({"error": f"Validation failed: {str(e)}", "valid": False}), 400

    def _setup_socket_handlers(self):
        """Setup WebSocket handlers for real-time updates"""
        
        if not self.socketio:
            logger.warning("SocketIO not available, skipping socket handlers")
            return

        @self.socketio.on('subscribe_job')
        def handle_job_subscription(data):
            """Subscribe to job updates"""
            job_id = data.get('job_id')
            if job_id:
                join_room(f"job_{job_id}")
                emit('subscribed', {'job_id': job_id})

        @self.socketio.on('unsubscribe_job')
        def handle_job_unsubscription(data):
            """Unsubscribe from job updates"""
            job_id = data.get('job_id')
            if job_id:
                leave_room(f"job_{job_id}")
                emit('unsubscribed', {'job_id': job_id})

    def _run_training_job(self, job_id: str):
        """Run training job in background"""
        try:
            job = self.job_manager.get_job(job_id)
            if not job:
                return

            # Create trainer
            config = UnifiedTrainingConfig.from_dict(job['config'])
            trainer = UnifiedTrainer(config)

            # Custom training loop with progress updates
            def progress_callback(epoch, metrics):
                progress = {
                    'current_epoch': epoch,
                    'current_loss': metrics.get('loss', 0),
                    'best_accuracy': metrics.get('accuracy', 0)
                }
                self.job_manager.update_job_progress(job_id, progress)
                
                # Broadcast update via WebSocket
                if self.socketio:
                    self.socketio.emit('job_update', {
                        'job_id': job_id,
                        'progress': progress,
                        'timestamp': datetime.now().isoformat()
                    }, room=f"job_{job_id}")

            # Run training with progress callback
            results = trainer.train(progress_callback=progress_callback)

            # Update job completion
            self.job_manager.update_job_progress(job_id, {
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'results': results
            })

            # Broadcast completion
            if self.socketio:
                self.socketio.emit('job_completed', {
                    'job_id': job_id,
                    'results': results
                }, room=f"job_{job_id}")

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            self.job_manager.update_job_progress(job_id, {
                'status': 'error',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            })

            # Broadcast error
            if self.socketio:
                self.socketio.emit('job_error', {
                    'job_id': job_id,
                    'error': str(e)
                }, room=f"job_{job_id}")

    def run(self, debug=None):
        """Start the enhanced API server"""
        debug = debug if debug is not None else self.debug
        logger.info(f"Starting Enhanced Train System Web API on {self.host}:{self.port}")
        
        if self.socketio:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)
        else:
            logger.warning("Running without WebSocket support")
            self.app.run(host=self.host, port=self.port, debug=debug)


def create_web_app():
    """Create Flask app for web deployment"""
    api = EnhancedTrainingAPI()
    return api.app


def run_web_server():
    """Entry point for enhanced web server"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Train System Web API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    api = EnhancedTrainingAPI(host=args.host, port=args.port, debug=args.debug)
    api.run()


if __name__ == "__main__":
    run_web_server()
