#!/usr/bin/env python3
"""
YOLO External Trainer using YOLO's built-in train function
This leverages YOLO's optimized training pipeline while integrating with train_system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from pathlib import Path
import logging
import os
import tempfile
import yaml

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_ERROR = None
except ImportError as e:
    YOLO_AVAILABLE = False
    YOLO_ERROR = f"ImportError: {e}"
    print("⚠️  Warning: ultralytics not available. Install with: pip install ultralytics")
except AttributeError as e:
    YOLO_AVAILABLE = False
    YOLO_ERROR = f"AttributeError (OpenCV compatibility issue): {e}"
    print("⚠️  Warning: OpenCV/ultralytics compatibility issue detected")
    print("   Try: pip install opencv-python==4.8.1.78 ultralytics==8.0.196")
except Exception as e:
    YOLO_AVAILABLE = False
    YOLO_ERROR = f"Unexpected error: {e}"
    print(f"⚠️  Warning: Failed to import ultralytics: {e}")

# Stub YOLO class for when ultralytics is not available
if not YOLO_AVAILABLE:
    class YOLO:
        def __init__(self, *args, **kwargs):
            raise ImportError("YOLO not available - this is a stub class")
        
        def train(self, *args, **kwargs):
            raise ImportError("YOLO not available - this is a stub class")


class YOLOBuiltinTrainer:
    """
    YOLO trainer that uses YOLO's built-in train() function
    Integrates YOLO's optimized training with train_system infrastructure
    """
    
    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        if not YOLO_AVAILABLE:
            # Provide clear error message with solutions
            error_msg = f"❌ YOLO trainer cannot be initialized: {YOLO_ERROR}\n"
            error_msg += "🔧 Possible solutions:\n"
            
            # Check if it's a NumPy 2.x compatibility issue
            if "numpy" in str(YOLO_ERROR).lower() and ("array_api" in str(YOLO_ERROR).lower() or "multiarray" in str(YOLO_ERROR).lower()):
                error_msg += "   🎯 NumPy 2.x compatibility issue detected:\n"
                error_msg += "   1. RECOMMENDED: pip install 'numpy<2'\n"
                error_msg += "   2. Or try: pip install opencv-python==4.8.1.78 ultralytics==8.0.196\n"
                error_msg += "   3. Alternative: pip install --upgrade ultralytics opencv-python\n"
                error_msg += "   4. Check environment: pip list | findstr /i numpy\n"
                error_msg += "   5. Clean install: pip uninstall numpy opencv-python ultralytics && pip install 'numpy<2' opencv-python ultralytics"
            else:
                error_msg += "   1. Fix NumPy compatibility: pip install 'numpy<2'\n"
                error_msg += "   2. Install compatible versions: pip install opencv-python==4.8.1.78 ultralytics==8.0.196\n"
                error_msg += "   3. Or try: pip install --upgrade ultralytics opencv-python\n"
                error_msg += "   4. Check for conflicting OpenCV installations: pip list | findstr /i opencv"
            
            raise ImportError(error_msg)
        
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
        
        # Initialize yolo_model as None first
        self.yolo_model = None
        
        # Create YOLO model instance
        self._setup_yolo_model()
        
        # Training state for compatibility
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        print(f"🎯 YOLOBuiltinTrainer initialized")
        print(f"   Using YOLO's built-in training pipeline")
        if self.yolo_model:
            print(f"   Model: {type(self.yolo_model).__name__}")
        else:
            print(f"   Model: None (YOLO model setup failed)")
        print(f"   Device: {device}")
        print(f"   Output: {output_dir}")
        print(f"   Parameters: {self.params}")
    
    def _setup_yolo_model(self):
        """Setup YOLO model for training"""
        # Get model architecture from config
        model_config = self.config.model
        
        if hasattr(model_config, 'path') and model_config.path:
            # Use custom model or pre-trained weights
            model_path = model_config.path
            if os.path.exists(model_path):
                print(f"   Loading YOLO model from: {model_path}")
                self.yolo_model = YOLO(model_path)
            else:
                # Use default YOLO model for classification
                print("   Using default YOLO classification model")
                self.yolo_model = YOLO('yolov8n-cls.pt')
        else:
            # Use default YOLO model
            print("   Using default YOLO classification model")
            self.yolo_model = YOLO('yolov8n-cls.pt')
        
        # If we have a trained model, load its weights
        if self.original_model is not None:
            try:
                # Extract state dict from train_system model
                if hasattr(self.original_model, 'state_dict'):
                    state_dict = self.original_model.state_dict()
                    # Try to load weights (may not be compatible)
                    self.yolo_model.model.load_state_dict(state_dict, strict=False)
                    print("   ✅ Loaded weights from train_system model")
            except Exception as e:
                print(f"   ⚠️  Could not load train_system weights: {e}")
                print("   Using YOLO pretrained weights instead")
    
    def _get_class_names(self):
        """Extract class names from dataset"""
        try:
            # Try to get class names from class_mapping
            if hasattr(self.config.data, 'class_mapping') and self.config.data.class_mapping:
                # Sort by class index to get correct order
                class_mapping = self.config.data.class_mapping
                sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
                return [name for name, idx in sorted_classes]
            
            # Try to get class names from data config
            if hasattr(self.config.data, 'class_names'):
                return self.config.data.class_names
            
            # Try to infer from train_loader
            if self.train_loader and hasattr(self.train_loader.dataset, 'classes'):
                return self.train_loader.dataset.classes
            
            # Default to numeric names
            num_classes = getattr(self.config.data, 'num_classes', self.config.model.num_classes)
            return [f'class_{i}' for i in range(num_classes)]
            
        except Exception as e:
            print(f"   ⚠️  Could not extract class names: {e}")
            # Fallback to binary classification
            return ['real', 'fake']
    
    def train(self) -> Dict[str, Any]:
        """Main training using YOLO's built-in train function"""
        print("🚀 Starting YOLO Built-in Training")
        print("=" * 60)
        
        if self.yolo_model is None:
            error_msg = "❌ Cannot start training: YOLO model is not initialized"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # Prepare training arguments directly from config
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
            
            print("✅ YOLO training completed!")
            print(f"   Results saved to: {self.output_dir}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ YOLO training failed: {e}")
            raise e
    
    def _prepare_train_args(self, dataset_yaml: str = None) -> Dict[str, Any]:
        """Prepare arguments for YOLO train function"""
        training_config = self.config.training
        data_config = self.config.data
        
        # Use train path directly instead of dataset.yaml
        train_path = str(Path(data_config.train_path).absolute())
        
        # Base arguments - use data path directly
        train_args = {
            'data': train_path,  # Direct path to training data
            'epochs': getattr(training_config, 'epochs', 50),
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
        
        # Add optimizer if specified
        optimizer = getattr(training_config, 'optimizer', 'adam').lower()
        if optimizer in ['adam', 'adamw', 'sgd']:
            train_args['optimizer'] = optimizer.upper()
        
        # Add scheduler if specified
        if hasattr(training_config, 'scheduler'):
            scheduler = training_config.scheduler.lower()
            if scheduler == 'cosine':
                train_args['cos_lr'] = True
            elif scheduler == 'step':
                train_args['lrf'] = 0.1  # Final learning rate factor
        
        # Add early stopping if configured
        if hasattr(training_config, 'early_stopping_patience'):
            train_args['patience'] = getattr(training_config, 'early_stopping_patience', 10)
        
        # Add custom parameters from external trainer config
        yolo_params = self.params.get('yolo_params', {})
        train_args.update(yolo_params)
        
        return train_args
    
    def _process_training_results(self, results) -> Dict[str, Any]:
        """Process YOLO training results into train_system format"""
        try:
            # Extract metrics from YOLO results
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                metrics = {}
            
            # Get best accuracy
            best_acc = 0.0
            if 'metrics/accuracy_top1' in metrics:
                best_acc = metrics['metrics/accuracy_top1']
            elif hasattr(results, 'best_fitness'):
                best_acc = results.best_fitness * 100  # Convert to percentage
            
            # Create train_system compatible results
            training_results = {
                'final_accuracy': best_acc,
                'best_val_accuracy': best_acc,  # For CLI compatibility
                'best_epoch': getattr(results, 'epoch', 0),
                'train_losses': getattr(results, 'train_losses', []),
                'val_accuracies': getattr(results, 'val_accuracies', [best_acc]),
                'yolo_results': metrics,
                'model_path': str(results.save_dir) if hasattr(results, 'save_dir') else str(self.output_dir)
            }
            
            # Update instance variables for compatibility
            self.best_acc = best_acc
            self.train_losses = training_results['train_losses']
            self.val_accuracies = training_results['val_accuracies']
            
            return training_results
            
        except Exception as e:
            print(f"⚠️  Warning: Could not extract detailed results: {e}")
            return {
                'final_accuracy': 0.0,
                'best_val_accuracy': 0.0,  # For CLI compatibility
                'best_epoch': 0,
                'train_losses': [],
                'val_accuracies': [],
                'status': 'completed'
            }
    
    def _save_train_system_checkpoint(self, training_results: Dict[str, Any]):
        """Save train_system compatible checkpoint"""
        try:
            # Get the trained YOLO model
            trained_model = self.yolo_model.model
            
            # Create train_system compatible checkpoint
            checkpoint = {
                'epoch': training_results['best_epoch'],
                'model_state_dict': trained_model.state_dict(),
                'best_val_acc': training_results['final_accuracy'],
                'train_losses': training_results['train_losses'],
                'val_accuracies': training_results['val_accuracies'],
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
                'yolo_results': training_results.get('yolo_results', {}),
                'training_method': 'yolo_builtin'
            }
            
            # Save train_system checkpoint
            torch.save(checkpoint, self.output_dir / 'best_checkpoint.pth')
            torch.save(checkpoint, self.output_dir / 'last_checkpoint.pth')
            
            print(f"   💾 Saved train_system checkpoint: best_checkpoint.pth")
            
            # The YOLO training already saves YOLO-compatible checkpoints
            # Copy the best one to our output directory for convenience
            yolo_output_dir = Path(training_results.get('model_path', self.output_dir))
            yolo_best_pt = yolo_output_dir / 'weights' / 'best.pt'
            
            if yolo_best_pt.exists():
                import shutil
                shutil.copy2(yolo_best_pt, self.output_dir / 'yolo_best.pt')
                print(f"   💾 Copied YOLO checkpoint: yolo_best.pt")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save train_system checkpoint: {e}")


if __name__ == "__main__":
    print("📄 YOLO External Trainer loaded")
    print("   Available trainer:")
    print("   - YOLOBuiltinTrainer: Uses YOLO's optimized built-in training")
