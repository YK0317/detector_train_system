#!/usr/bin/env python3
"""
Robust YOLO External Trainer that handles import issues gracefully
This version delays imports and provides fallbacks for compatibility issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import os
import tempfile
import yaml
import sys
import importlib

# Global variables to track import status
YOLO_IMPORT_ERROR = None
YOLO_CLASS = None


def safe_import_yolo():
    """Safely import YOLO with detailed error reporting"""
    global YOLO_IMPORT_ERROR, YOLO_CLASS
    
    if YOLO_CLASS is not None:
        return YOLO_CLASS
    
    if YOLO_IMPORT_ERROR is not None:
        raise YOLO_IMPORT_ERROR
    
    try:
        # Try importing step by step to isolate issues
        print("🔍 Attempting to import YOLO components...")
        
        # Step 1: Import cv2
        print("   Step 1: Importing cv2...")
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__} imported successfully")
        
        # Step 2: Import ultralytics
        print("   Step 2: Importing ultralytics...")
        import ultralytics
        print(f"   ✅ Ultralytics {ultralytics.__version__} imported successfully")
        
        # Step 3: Import YOLO class
        print("   Step 3: Importing YOLO class...")
        from ultralytics import YOLO
        print("   ✅ YOLO class imported successfully")
        
        YOLO_CLASS = YOLO
        return YOLO_CLASS
        
    except Exception as e:
        error_msg = f"Failed to import YOLO: {e}"
        print(f"   ❌ {error_msg}")
        YOLO_IMPORT_ERROR = ImportError(error_msg)
        raise YOLO_IMPORT_ERROR


class YOLOBuiltinTrainerRobust:
    """
    Robust YOLO trainer that handles import and compatibility issues
    """
    
    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        self.config = config
        self.original_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Get external trainer parameters
        external_params = getattr(config, 'external_trainer', {})
        if hasattr(external_params, 'parameters'):
            self.params = external_params.parameters
        else:
            self.params = {}
        
        # YOLO model will be initialized lazily
        self.yolo_model = None
        
        # Training state for compatibility
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        print(f"🎯 YOLOBuiltinTrainerRobust initialized")
        print(f"   Device: {device}")
        print(f"   Output: {output_dir}")
        print(f"   Parameters: {self.params}")
        
        # Test YOLO import but don't fail here - delay until training
        try:
            safe_import_yolo()
            print("   ✅ YOLO import test successful")
        except Exception as e:
            print(f"   ⚠️ YOLO import test failed: {e}")
            print("   Will retry during training...")
    
    def _setup_yolo_model(self):
        """Setup YOLO model for training with robust error handling"""
        if self.yolo_model is not None:
            return
            
        try:
            # Import YOLO with error handling
            YOLO = safe_import_yolo()
            
            # Get model configuration
            model_config = self.config.model
            
            # Determine model path/name
            model_name = "yolov8n-cls.pt"  # Default
            
            if hasattr(model_config, 'model_args') and model_config.model_args:
                if hasattr(model_config.model_args, 'model_name'):
                    model_name = model_config.model_args.model_name
            
            if hasattr(model_config, 'path') and model_config.path:
                if os.path.exists(model_config.path):
                    model_name = model_config.path
                    print(f"   Using custom model: {model_name}")
                else:
                    print(f"   Model path {model_config.path} not found, using default")
            
            print(f"   Creating YOLO model: {model_name}")
            self.yolo_model = YOLO(model_name)
            print("   ✅ YOLO model created successfully")
            
        except Exception as e:
            error_msg = f"Failed to setup YOLO model: {e}"
            self.logger.error(error_msg)
            print(f"   ❌ {error_msg}")
            
            # Provide a helpful error message
            self._print_troubleshooting_guide()
            raise RuntimeError(error_msg)
    
    def _print_troubleshooting_guide(self):
        """Print troubleshooting guide for common issues"""
        print("\n" + "="*60)
        print("🔧 TROUBLESHOOTING GUIDE")
        print("="*60)
        print("If you're seeing import errors, try these solutions:")
        print()
        print("1. OpenCV/NumPy compatibility issues:")
        print("   pip uninstall opencv-python opencv-python-headless -y")
        print("   pip install opencv-python==4.8.1.78")
        print()
        print("2. NumPy version conflicts:")
        print("   pip install 'numpy<2.0'")
        print()
        print("3. Ultralytics installation:")
        print("   pip install ultralytics --upgrade")
        print()
        print("4. Complete clean install:")
        print("   pip uninstall ultralytics opencv-python -y")
        print("   pip install ultralytics")
        print()
        print("Current environment:")
        print(f"   Python: {sys.version}")
        
        try:
            import numpy as np
            print(f"   NumPy: {np.__version__}")
        except:
            print("   NumPy: Not available")
            
        try:
            import cv2
            print(f"   OpenCV: {cv2.__version__}")
        except:
            print("   OpenCV: Not available")
            
        print("="*60)
    
    def train(self) -> Dict[str, Any]:
        """Main training method with robust error handling"""
        print("🚀 Starting Robust YOLO Training")
        print("=" * 60)
        
        try:
            # Setup YOLO model (this is where most errors occur)
            self._setup_yolo_model()
            
            # Prepare training arguments
            train_args = self._prepare_train_args()
            
            print(f"📋 YOLO Training Arguments:")
            for key, value in train_args.items():
                print(f"   {key}: {value}")
            
            # Run YOLO training
            print(f"\n🏃 Starting YOLO training...")
            results = self.yolo_model.train(**train_args)
            
            # Process results
            training_results = self._process_training_results(results)
            
            # Save train_system compatible checkpoint
            self._save_train_system_checkpoint(training_results)
            
            print("✅ YOLO training completed successfully!")
            print(f"   Results saved to: {self.output_dir}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ YOLO training failed: {e}")
            print(f"\n❌ Training failed: {e}")
            
            # Provide fallback result
            fallback_result = {
                'success': False,
                'error': str(e),
                'best_accuracy': 0.0,
                'final_loss': float('inf'),
                'epochs_completed': 0,
                'message': 'Training failed due to YOLO import/setup issues'
            }
            
            return fallback_result
    
    def _prepare_train_args(self) -> Dict[str, Any]:
        """Prepare arguments for YOLO train function"""
        training_config = self.config.training
        data_config = self.config.data
        
        # Use train path directly
        train_path = str(Path(data_config.train_path).absolute())
        
        # Base training arguments
        train_args = {
            'data': train_path,
            'epochs': getattr(training_config, 'epochs', 10),  # Reduced default for testing
            'imgsz': getattr(data_config, 'img_size', 224),
            'batch': getattr(data_config, 'batch_size', 16),
            'lr0': getattr(training_config, 'learning_rate', 0.001),
            'weight_decay': getattr(training_config, 'weight_decay', 0.0005),
            'device': str(self.device),
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,
            'save': True,
            'save_period': getattr(training_config, 'save_frequency', 5),
            'workers': getattr(data_config, 'num_workers', 4),
            'verbose': True
        }
        
        # Add validation path if specified
        if hasattr(data_config, 'val_path') and data_config.val_path:
            train_args['val'] = str(Path(data_config.val_path).absolute())
        
        # Add YOLO-specific parameters
        if hasattr(self.params, 'yolo_params') and self.params.yolo_params:
            yolo_params = self.params.yolo_params
            for key, value in yolo_params.items():
                if value is not None:
                    train_args[key] = value
        
        return train_args
    
    def _process_training_results(self, results) -> Dict[str, Any]:
        """Process YOLO training results"""
        try:
            # Extract metrics from YOLO results
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                metrics = {}
            
            return {
                'success': True,
                'best_accuracy': metrics.get('metrics/accuracy_top1', 0.0),
                'final_loss': metrics.get('train/loss', 0.0),
                'epochs_completed': metrics.get('epoch', 0),
                'yolo_results': results,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"⚠️ Could not process results: {e}")
            return {
                'success': True,
                'best_accuracy': 0.0,
                'final_loss': 0.0,
                'epochs_completed': 0,
                'yolo_results': results,
                'message': 'Training completed but could not extract detailed metrics'
            }
    
    def _save_train_system_checkpoint(self, results: Dict[str, Any]):
        """Save a checkpoint compatible with train_system"""
        try:
            checkpoint_path = self.output_dir / "train_system_checkpoint.pth"
            
            checkpoint = {
                'model_state_dict': {},  # YOLO handles its own model saving
                'optimizer_state_dict': {},
                'scheduler_state_dict': {},
                'epoch': results.get('epochs_completed', 0),
                'best_acc': results.get('best_accuracy', 0.0),
                'train_losses': [results.get('final_loss', 0.0)],
                'val_accuracies': [results.get('best_accuracy', 0.0)],
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                'yolo_results': results
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"   ✅ Train system checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            print(f"   ⚠️ Could not save train_system checkpoint: {e}")


# Registry name for auto-discovery
YOLOBuiltinTrainer = YOLOBuiltinTrainerRobust
