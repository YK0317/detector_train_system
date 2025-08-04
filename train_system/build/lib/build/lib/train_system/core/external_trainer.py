#!/usr/bin/env python3
"""
External Training Method Override System
Allows users to specify custom training scripts from config file
"""

import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import TrainingConfig


class ExternalTrainerInterface:
    """
    Base interface that external training methods should implement
    This provides a contract for what methods external trainers need to have
    """
    
    def __init__(self, config: TrainingConfig, model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 device: torch.device, output_dir: Path):
        """
        Initialize external trainer
        
        Args:
            config: Training configuration
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            device: Device to train on
            output_dir: Directory to save outputs
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
    
    def train(self) -> Dict[str, Any]:
        """
        Main training method - must be implemented by external trainers
        
        Returns:
            Dict containing training results/metrics
        """
        raise NotImplementedError("External trainers must implement train() method")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save checkpoint - optional override for custom saving logic
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        }
        
        # Save current checkpoint
        torch.save(checkpoint, self.output_dir / 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_checkpoint.pth')


class ExternalTrainerLoader:
    """
    Loads and manages external training methods from user-specified scripts
    """
    
    @staticmethod
    def load_external_trainer(script_path: str, class_name: str = None) -> type:
        """
        Load external trainer class from script file
        
        Args:
            script_path: Path to Python script containing training method
            class_name: Name of trainer class (auto-detect if None)
            
        Returns:
            External trainer class
        """
        script_path = Path(script_path)
        
        if not script_path.exists():
            raise FileNotFoundError(f"External training script not found: {script_path}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("external_trainer", script_path)
        module = importlib.util.module_from_spec(spec)
        
        # Add script directory to path for relative imports
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Failed to load external training script: {e}")
        
        # Find trainer class
        trainer_class = None
        
        if class_name:
            # Use specified class name
            if hasattr(module, class_name):
                trainer_class = getattr(module, class_name)
            else:
                raise AttributeError(f"Class '{class_name}' not found in {script_path}")
        else:
            # Auto-detect trainer class
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    obj.__module__ == module.__name__ and
                    ('trainer' in name.lower() or 'training' in name.lower())):
                    trainer_class = obj
                    break
        
        if trainer_class is None:
            raise ValueError(f"No suitable trainer class found in {script_path}")
        
        # Validate interface
        ExternalTrainerLoader._validate_trainer_interface(trainer_class)
        
        return trainer_class
    
    @staticmethod
    def _validate_trainer_interface(trainer_class: type):
        """
        Validate that external trainer implements required interface
        
        Args:
            trainer_class: External trainer class to validate
        """
        required_methods = ['__init__', 'train']
        
        for method in required_methods:
            if not hasattr(trainer_class, method):
                raise ValueError(f"External trainer must implement '{method}' method")
        
        # Check __init__ signature
        init_sig = inspect.signature(trainer_class.__init__)
        required_params = ['config', 'model', 'train_loader', 'val_loader', 'device', 'output_dir']
        
        for param in required_params:
            if param not in init_sig.parameters:
                print(f"⚠️  Warning: External trainer __init__ missing parameter: {param}")
        
        # Check train method signature
        train_sig = inspect.signature(trainer_class.train)
        if len(train_sig.parameters) < 1:  # self parameter
            raise ValueError("External trainer 'train' method should accept self parameter")


class HybridTrainer:
    """
    Hybrid trainer that can use either built-in or external training methods
    """
    
    def __init__(self, config: TrainingConfig, model: Optional[nn.Module], device: torch.device, output_dir: Path):
        self.config = config
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Determine training method first
        self.external_trainer = None
        self.use_external = self._should_use_external_trainer()
        
        if self.use_external:
            # Check if external trainer handles its own data loading
            if self._external_trainer_handles_data():
                # Skip data loader creation for trainers like YOLO that handle their own data
                self.train_loader, self.val_loader = None, None
            else:
                # Create data loaders for external trainers that need them
                self.train_loader, self.val_loader = self._create_data_loaders()
            self._load_external_trainer()
        else:
            # Built-in training requires a model
            if self.model is None:
                raise ValueError("Model is required for built-in training")
            # Create data loaders for built-in training
            self.train_loader, self.val_loader = self._create_data_loaders()
    
    def _create_data_loaders(self):
        """Create data loaders from config"""
        from .dataset import UnifiedDataset
        import torchvision.transforms as transforms
        
        # Create transforms (simplified version)
        img_size = getattr(self.config.data, 'img_size', 224)
        
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets using config-based control
        train_dataset = UnifiedDataset(self.config.data, 'train', train_transform)
        val_dataset = UnifiedDataset(self.config.data, 'val', val_transform)
        
        # Create data loaders
        batch_size = getattr(self.config.data, 'batch_size', 16)
        num_workers = getattr(self.config.data, 'num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def _should_use_external_trainer(self) -> bool:
        """Check if external trainer should be used based on config"""
        if not hasattr(self.config, 'external_trainer'):
            return False
        
        external_config = getattr(self.config, 'external_trainer', {})
        
        # Handle both dict and ExternalTrainerConfig object
        if isinstance(external_config, dict):
            return external_config.get('enabled', False)
        elif hasattr(external_config, 'enabled'):
            return external_config.enabled
        
        return False
    
    def _external_trainer_handles_data(self) -> bool:
        """Check if external trainer handles its own data loading"""
        external_config = getattr(self.config, 'external_trainer', {})
        
        # Check the explicit configuration flag
        if isinstance(external_config, dict):
            return external_config.get('skip_unified_data_loading', False)
        elif hasattr(external_config, 'skip_unified_data_loading'):
            return external_config.skip_unified_data_loading
        
        return False
    
    def _load_external_trainer(self):
        """Load external trainer from config"""
        external_config = getattr(self.config, 'external_trainer', {})
        
        # Handle both dict and ExternalTrainerConfig object
        if isinstance(external_config, dict):
            script_path = external_config.get('script_path')
            class_name = external_config.get('class_name')
            name = external_config.get('name')
        else:
            script_path = getattr(external_config, 'script_path', None)
            class_name = getattr(external_config, 'class_name', None)
            name = getattr(external_config, 'name', None)
        
        # Try registry lookup first if name is provided
        if name:
            try:
                from ..registry import get_component_by_name
                TrainerClass = get_component_by_name('trainer', name)
                if TrainerClass:
                    print(f"🔄 Loading external trainer from registry: {name}")
                    
                    # Check if this is an ultralytics YOLO class or our custom trainer
                    module_name = getattr(TrainerClass, '__module__', '')
                    if 'ultralytics' in module_name:
                        print(f"⚠️  Detected ultralytics YOLO class, which should not be used as trainer")
                        print(f"   Please use 'yolobuiltin' instead of 'yolo' for proper YOLO training")
                        # Fallback to the YOLOBuiltinTrainer
                        try:
                            from ..registry import get_component_by_name
                            TrainerClass = get_component_by_name('trainer', 'yolobuiltin')
                            if not TrainerClass:
                                raise ValueError(f"YOLOBuiltinTrainer not found in registry")
                            print(f"🔄 Falling back to YOLOBuiltinTrainer")
                        except Exception as e:
                            raise ValueError(f"Cannot load YOLOBuiltinTrainer: {e}")
                    
                    # Try to initialize with our standard trainer interface
                    try:
                        self.external_trainer = TrainerClass(
                            config=self.config,
                            model=self.model,
                            train_loader=self.train_loader,
                            val_loader=self.val_loader,
                            device=self.device,
                            output_dir=self.output_dir
                        )
                    except TypeError as e:
                        if "unexpected keyword argument" in str(e):
                            print(f"⚠️  Trainer {TrainerClass.__name__} doesn't follow standard interface")
                            print(f"   Error: {e}")
                            raise ValueError(f"Trainer '{name}' has incompatible constructor signature: {e}")
                        else:
                            raise
                    
                    print(f"✅ External trainer loaded from registry: {TrainerClass.__name__}")
                    return
                else:
                    print(f"⚠️  Trainer '{name}' not found in registry, falling back to script_path")
            except ImportError:
                print("⚠️  Registry system not available, falling back to script_path")
        
        # Fallback to script_path loading
        if not script_path:
            if name:
                raise ValueError(f"External trainer '{name}' not found in registry and no script_path specified")
            else:
                raise ValueError("External trainer enabled but no name or script_path specified")
        
        print(f"🔄 Loading external trainer from: {script_path}")
        
        # Load trainer class
        TrainerClass = ExternalTrainerLoader.load_external_trainer(script_path, class_name)
        
        # Initialize external trainer with appropriate parameters
        self.external_trainer = TrainerClass(
            config=self.config,
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            output_dir=self.output_dir
        )
        
        print(f"✅ External trainer loaded: {TrainerClass.__name__}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training method - delegates to external or built-in trainer
        
        Returns:
            Training results
        """
        if self.use_external and self.external_trainer:
            print("🚀 Starting training with external trainer")
            return self.external_trainer.train()
        else:
            print("🚀 Starting training with built-in trainer")
            return self._builtin_train()
    
    def _builtin_train(self) -> Dict[str, Any]:
        """
        Built-in training method (fallback)
        This is a simplified version - you'd use your existing UnifiedTrainer logic
        """
        from .trainer import UnifiedTrainer
        
        # Create unified trainer instance
        trainer = UnifiedTrainer(self.config, self.model, self.device, self.output_dir)
        
        # Run training
        return trainer.train()


def create_example_external_trainer():
    """Create an example external trainer script for users"""
    example_code = '''#!/usr/bin/env python3
"""
Example External Training Method
This shows how to create a custom training method for the train_system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from pathlib import Path

class CustomYOLOTrainer:
    """
    Example custom trainer for YOLO models
    """
    
    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Setup optimizer and criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
    
    def train(self) -> Dict[str, Any]:
        """Custom training loop"""
        print("🎯 Starting Custom YOLO Training")
        
        epochs = getattr(self.config.training, 'epochs', 10)
        
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self._train_epoch()
            
            # Validate
            val_acc = self._validate()
            
            # Save checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            
            self.save_checkpoint(epoch, {'loss': train_loss, 'accuracy': val_acc}, is_best)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={val_acc:.2f}%")
        
        return {
            'final_accuracy': self.best_acc,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def _train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def _validate(self) -> float:
        """Validate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        self.val_accuracies.append(accuracy)
        return accuracy
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint with YOLO-compatible format"""
        
        # Standard checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else {}
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.output_dir / 'last_checkpoint.pth')
        
        if is_best:
            # Save best checkpoint in train_system format
            torch.save(checkpoint, self.output_dir / 'best_checkpoint.pth')
            
            # Also save YOLO-compatible format
            yolo_checkpoint = {
                'model': self.model,  # Full model for YOLO compatibility
                'epoch': epoch,
                'best_fitness': self.best_acc / 100.0,
                'date': '2025-01-01',
                'version': '8.0.0',
                'train_metrics': {
                    'metrics/accuracy_top1': self.best_acc,
                    'val/loss': metrics.get('loss', 0.0)
                }
            }
            torch.save(yolo_checkpoint, self.output_dir / 'yolo_compatible.pt')
            print(f"✅ Saved YOLO-compatible checkpoint: yolo_compatible.pt")


# Alternative trainer example
class SimpleCustomTrainer:
    """Minimal custom trainer example"""
    
    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        self.config = config
        self.model = model.to(device)
        self.device = device
        print(f"🎯 Custom trainer initialized with {len(train_loader)} training batches")
    
    def train(self):
        """Simple training implementation"""
        print("🚀 Running simple custom training...")
        
        # Your custom training logic here
        epochs = 5  # Simple example
        
        for epoch in range(epochs):
            print(f"Custom epoch {epoch+1}/{epochs}")
            # Add your training code here
        
        return {'status': 'completed', 'method': 'custom'}
'''
    
    return example_code


if __name__ == "__main__":
    # Create example file
    example_path = Path("example_external_trainer.py")
    with open(example_path, 'w') as f:
        f.write(create_example_external_trainer())
    
    print(f"✅ Example external trainer created: {example_path}")
