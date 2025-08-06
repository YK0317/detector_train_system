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

# Robust YOLO import with fallback
YOLO_AVAILABLE = False
YOLO = None


def import_yolo_safely():
    """Safely import YOLO with error handling"""
    global YOLO_AVAILABLE, YOLO

    if YOLO_AVAILABLE and YOLO is not None:
        return True

    try:
        # Try to disable OpenCV G-API to avoid typing issues
        os.environ["OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019"] = "0"
        os.environ["OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NGRAPH"] = "0"

        # Import ultralytics
        from ultralytics import YOLO as _YOLO

        YOLO = _YOLO
        YOLO_AVAILABLE = True
        print("✅ YOLO imported successfully")
        return True

    except Exception as e:
        print(f"❌ YOLO import failed: {e}")
        print("🔧 Trying fallback import...")

        try:
            # Try alternative import approach
            import sys

            if "cv2" in sys.modules:
                del sys.modules["cv2"]

            # Disable problematic cv2 features
            os.environ["OPENCV_DNN_BACKEND_DEFAULT"] = "0"

            from ultralytics import YOLO as _YOLO

            YOLO = _YOLO
            YOLO_AVAILABLE = True
            print("✅ YOLO imported with fallback method")
            return True

        except Exception as e2:
            print(f"❌ Fallback import also failed: {e2}")
            print(
                "💡 Suggestion: Try 'pip install opencv-python==4.8.1.78 ultralytics==8.0.196'"
            )
            YOLO_AVAILABLE = False
            return False


class YOLOBuiltinTrainer:
    """
    YOLO trainer that uses YOLO's built-in train() function
    Integrates YOLO's optimized training with train_system infrastructure
    """

    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        # Try to import YOLO when actually needed
        if not import_yolo_safely():
            raise ImportError(
                "❌ YOLO import failed. This is likely due to OpenCV compatibility issues.\n"
                "🔧 Try these solutions:\n"
                "   1. pip uninstall opencv-python opencv-python-headless -y\n"
                "   2. pip install opencv-python==4.8.1.78\n"
                "   3. pip install ultralytics==8.0.196\n"
                "   4. Or use conda: conda install opencv=4.8.1 ultralytics=8.0.196"
            )

        self.config = config
        self.original_model = model  # This might be a train_system model for dry-run
        self.device = device
        self.output_dir = Path(output_dir)

        # For YOLO trainers that skip unified data loading, we need to create minimal data loaders
        # for compatibility with dry-run and external trainer interface
        if train_loader is None or val_loader is None:
            print(
                "   🔧 Creating minimal data loaders for YOLO trainer compatibility..."
            )
            self.train_loader, self.val_loader = self._create_minimal_data_loaders()
        else:
            self.train_loader = train_loader
            self.val_loader = val_loader

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Get external trainer parameters
        external_params = getattr(config, "external_trainer", {})
        if hasattr(external_params, "parameters"):
            self.params = external_params.parameters
        else:
            self.params = {}

        # Create YOLO model instance (now that we know import works)
        self._setup_yolo_model()

        # Training state for compatibility
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []

        print(f"🎯 YOLOBuiltinTrainer initialized")
        print(f"   Using YOLO's built-in training pipeline")
        print(f"   Model: {type(self.yolo_model).__name__}")
        print(f"   Device: {device}")
        print(f"   Output: {output_dir}")
        print(f"   Parameters: {self.params}")

        # If this is a dry-run with train_system model, we'll note it
        if self.original_model is not None:
            print(
                f"   🧪 Dry-run mode: Received train_system model {type(self.original_model).__name__}"
            )
            print(f"   🔄 Will use YOLO model for actual training")

    def _create_minimal_data_loaders(self):
        """Create minimal data loaders for dry-run compatibility"""
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            from PIL import Image
            import os
            from pathlib import Path

            class MinimalDataset(Dataset):
                def __init__(self, data_path, max_samples=5):
                    self.data_path = Path(data_path)
                    self.samples = []
                    self.max_samples = max_samples

                    # Look for class folders (real/fake structure)
                    train_path = (
                        self.data_path / "train"
                        if (self.data_path / "train").exists()
                        else self.data_path
                    )

                    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]

                    sample_count = 0
                    for class_idx, class_dir in enumerate(class_dirs):
                        if sample_count >= self.max_samples:
                            break

                        for img_file in class_dir.glob("*"):
                            if img_file.suffix.lower() in [
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".bmp",
                            ]:
                                self.samples.append((str(img_file), class_idx))
                                sample_count += 1
                                if sample_count >= self.max_samples:
                                    break

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    # Create dummy tensor data for dry-run
                    dummy_img = torch.randn(3, 224, 224)  # RGB image
                    return dummy_img, torch.tensor(label, dtype=torch.long)

            # Get data path from config
            data_config = self.config.data
            train_path = getattr(data_config, "train_path", "")

            if train_path and os.path.exists(train_path):
                train_dataset = MinimalDataset(train_path, max_samples=5)
                val_dataset = MinimalDataset(
                    train_path, max_samples=2
                )  # Smaller for validation

                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

                print(
                    f"      Created minimal data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples"
                )
                return train_loader, val_loader
            else:
                # Create completely dummy data loaders
                print(f"      Creating dummy data loaders for dry-run")

                class DummyDataset(Dataset):
                    def __init__(self, size=5):
                        self.size = size

                    def __len__(self):
                        return self.size

                    def __getitem__(self, idx):
                        return torch.randn(3, 224, 224), torch.tensor(
                            idx % 2, dtype=torch.long
                        )

                train_dataset = DummyDataset(5)
                val_dataset = DummyDataset(2)

                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

                return train_loader, val_loader

        except Exception as e:
            print(f"      ⚠️  Failed to create minimal data loaders: {e}")
            # Return None if we can't create loaders
            return None, None

    def _setup_yolo_model(self):
        """Setup YOLO model for training"""
        global YOLO

        # Get model architecture from config
        model_config = self.config.model

        try:
            if hasattr(model_config, "path") and model_config.path:
                # Use custom model or pre-trained weights
                model_path = model_config.path
                if os.path.exists(model_path):
                    print(f"   Loading YOLO model from: {model_path}")
                    self.yolo_model = YOLO(model_path)
                else:
                    # Use default YOLO model for classification
                    print("   Using default YOLO classification model")
                    self.yolo_model = YOLO("yolov8n-cls.pt")
            else:
                # Use default YOLO model
                print("   Using default YOLO classification model")
                self.yolo_model = YOLO("yolov8n-cls.pt")
        except Exception as e:
            print(f"❌ YOLO model setup failed: {e}")
            print("🔧 Trying alternative model...")
            try:
                # Try with a smaller model that might work better
                self.yolo_model = YOLO("yolov8n.pt")  # Detection model as fallback
                print("   ✅ Using YOLO detection model as fallback")
            except Exception as e2:
                raise RuntimeError(f"Failed to setup any YOLO model: {e2}")

        # If we have a trained model, load its weights
        if self.original_model is not None:
            try:
                # Extract state dict from train_system model
                if hasattr(self.original_model, "state_dict"):
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
            if (
                hasattr(self.config.data, "class_mapping")
                and self.config.data.class_mapping
            ):
                # Sort by class index to get correct order
                class_mapping = self.config.data.class_mapping
                sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
                return [name for name, idx in sorted_classes]

            # Try to get class names from data config
            if hasattr(self.config.data, "class_names"):
                return self.config.data.class_names

            # Try to infer from train_loader
            if self.train_loader and hasattr(self.train_loader.dataset, "classes"):
                return self.train_loader.dataset.classes

            # Default to numeric names
            num_classes = getattr(
                self.config.data, "num_classes", self.config.model.num_classes
            )
            return [f"class_{i}" for i in range(num_classes)]

        except Exception as e:
            print(f"   ⚠️  Could not extract class names: {e}")
            # Fallback to binary classification
            return ["real", "fake"]

    def train(self) -> Dict[str, Any]:
        """Main training using YOLO's built-in train function"""
        print("🚀 Starting YOLO Built-in Training")
        print("=" * 60)

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
            "data": train_path,  # Direct path to training data
            "epochs": getattr(training_config, "epochs", 50),
            "imgsz": getattr(data_config, "img_size", 224),
            "batch": getattr(data_config, "batch_size", 16),
            "lr0": getattr(training_config, "learning_rate", 0.001),
            "weight_decay": getattr(training_config, "weight_decay", 0.0005),
            "device": str(self.device),
            "project": str(self.output_dir.parent),
            "name": self.output_dir.name,
            "exist_ok": True,
            "save": True,
            "save_period": getattr(training_config, "save_frequency", 5),
            "workers": getattr(data_config, "num_workers", 4),
            "verbose": True,
        }

        # Add validation path if specified
        if hasattr(data_config, "val_path") and data_config.val_path:
            train_args["val"] = str(Path(data_config.val_path).absolute())

        # Add optimizer if specified
        optimizer = getattr(training_config, "optimizer", "adam").lower()
        if optimizer in ["adam", "adamw", "sgd"]:
            train_args["optimizer"] = optimizer.upper()

        # Add scheduler if specified
        if hasattr(training_config, "scheduler"):
            scheduler = training_config.scheduler.lower()
            if scheduler == "cosine":
                train_args["cos_lr"] = True
            elif scheduler == "step":
                train_args["lrf"] = 0.1  # Final learning rate factor

        # Add early stopping if configured
        if hasattr(training_config, "early_stopping_patience"):
            train_args["patience"] = getattr(
                training_config, "early_stopping_patience", 10
            )

        # Add custom parameters from external trainer config
        yolo_params = self.params.get("yolo_params", {})
        train_args.update(yolo_params)

        return train_args

    def _process_training_results(self, results) -> Dict[str, Any]:
        """Process YOLO training results into train_system format"""
        try:
            # Extract metrics from YOLO results
            if hasattr(results, "results_dict"):
                metrics = results.results_dict
            else:
                metrics = {}

            # Get best accuracy
            best_acc = 0.0
            if "metrics/accuracy_top1" in metrics:
                best_acc = metrics["metrics/accuracy_top1"]
            elif hasattr(results, "best_fitness"):
                best_acc = results.best_fitness * 100  # Convert to percentage

            # Create train_system compatible results
            training_results = {
                "final_accuracy": best_acc,
                "best_epoch": getattr(results, "epoch", 0),
                "train_losses": getattr(results, "train_losses", []),
                "val_accuracies": getattr(results, "val_accuracies", [best_acc]),
                "yolo_results": metrics,
                "model_path": (
                    str(results.save_dir)
                    if hasattr(results, "save_dir")
                    else str(self.output_dir)
                ),
            }

            # Update instance variables for compatibility
            self.best_acc = best_acc
            self.train_losses = training_results["train_losses"]
            self.val_accuracies = training_results["val_accuracies"]

            return training_results

        except Exception as e:
            print(f"⚠️  Warning: Could not extract detailed results: {e}")
            return {
                "final_accuracy": 0.0,
                "best_epoch": 0,
                "train_losses": [],
                "val_accuracies": [],
                "status": "completed",
            }

    def _save_train_system_checkpoint(self, training_results: Dict[str, Any]):
        """Save train_system compatible checkpoint"""
        try:
            # Get the trained YOLO model
            trained_model = self.yolo_model.model

            # Create train_system compatible checkpoint
            checkpoint = {
                "epoch": training_results["best_epoch"],
                "model_state_dict": trained_model.state_dict(),
                "best_val_acc": training_results["final_accuracy"],
                "train_losses": training_results["train_losses"],
                "val_accuracies": training_results["val_accuracies"],
                "config": (
                    self.config.to_dict()
                    if hasattr(self.config, "to_dict")
                    else vars(self.config)
                ),
                "yolo_results": training_results.get("yolo_results", {}),
                "training_method": "yolo_builtin",
            }

            # Save train_system checkpoint
            torch.save(checkpoint, self.output_dir / "best_checkpoint.pth")
            torch.save(checkpoint, self.output_dir / "last_checkpoint.pth")

            print(f"   💾 Saved train_system checkpoint: best_checkpoint.pth")

            # The YOLO training already saves YOLO-compatible checkpoints
            # Copy the best one to our output directory for convenience
            yolo_output_dir = Path(training_results.get("model_path", self.output_dir))
            yolo_best_pt = yolo_output_dir / "weights" / "best.pt"

            if yolo_best_pt.exists():
                import shutil

                shutil.copy2(yolo_best_pt, self.output_dir / "yolo_best.pt")
                print(f"   💾 Copied YOLO checkpoint: yolo_best.pt")

        except Exception as e:
            print(f"⚠️  Warning: Could not save train_system checkpoint: {e}")


# Alternative simpler trainer for demonstration
class SimpleExternalTrainer:
    """Minimal external trainer example"""

    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)

        print(f"🎯 SimpleExternalTrainer initialized")

    def train(self):
        """Simple training implementation"""
        print("🚀 Running simple external training...")

        epochs = getattr(self.config.training, "epochs", 5)

        for epoch in range(epochs):
            print(f"   Custom epoch {epoch+1}/{epochs}")
            # Your simple training logic here

        print("✅ Simple training completed!")

        return {"status": "completed", "method": "simple_external", "epochs": epochs}


if __name__ == "__main__":
    print("📄 External trainer example loaded")
    print("   Available trainers:")
    print("   - YOLOBuiltinTrainer: Uses YOLO's optimized built-in training")
    print("   - CustomYOLOTrainer: Custom training loop with YOLO-compatible saving")
    print("   - SimpleExternalTrainer: Minimal example trainer")
