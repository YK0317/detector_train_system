#!/usr/bin/env python3
"""
Capsule-Forensics External Trainer

This trainer integrates the original Capsule-Forensics-v2 training logic
with the train_system infrastructure, allowing users to leverage the
exact training methodology from the research paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from typing import Dict, Any
from pathlib import Path
import logging
import os
import sys
import tempfile
import shutil
import importlib.util
from tqdm import tqdm
import numpy as np
from sklearn import metrics


class CapsuleForensicsTrainer:
    """
    External trainer that uses original Capsule-Forensics-v2 training logic
    Integrates with train_system while preserving the exact methodology
    """

    def __init__(self, config, model, train_loader, val_loader, device, output_dir):
        self.config = config
        self.original_model = model  # This might be None if using external trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)

        # Validate that we have data loaders
        if self.train_loader is None:
            raise ValueError(
                "Train loader is None. Please check your dataset paths in the configuration."
            )
        if self.val_loader is None:
            raise ValueError(
                "Validation loader is None. Please check your dataset paths in the configuration."
            )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Get external trainer parameters
        external_params = getattr(config, "external_trainer", {})
        if hasattr(external_params, "parameters"):
            self.params = external_params.parameters
        else:
            self.params = {}

        # Import Capsule-Forensics model components
        self._import_capsule_modules()

        # Initialize Capsule-Forensics components
        self._setup_capsule_models()

        # Training state
        self.best_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []

        print(f"🎯 CapsuleForensicsTrainer initialized")
        print(f"   Using original Capsule-Forensics-v2 training methodology")
        print(f"   Device: {device}")
        print(f"   Output: {output_dir}")
        print(f"   Resume epoch: {self.params.get('resume_epoch', 0)}")

    def _import_capsule_modules(self):
        """Import Capsule-Forensics model modules"""
        try:
            # Add Capsule-Forensics-v2 to Python path
            capsule_path = Path("Capsule-Forensics-v2")
            if not capsule_path.exists():
                # Try relative path
                capsule_path = Path("../Capsule-Forensics-v2")

            if not capsule_path.exists():
                raise FileNotFoundError("Capsule-Forensics-v2 directory not found")

            # Import model_big module
            spec = importlib.util.spec_from_file_location(
                "model_big", capsule_path / "model_big.py"
            )
            self.model_big = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.model_big)

            print(f"   ✅ Imported Capsule-Forensics modules from: {capsule_path}")

        except Exception as e:
            raise ImportError(f"Failed to import Capsule-Forensics modules: {e}")

    def _setup_capsule_models(self):
        """Setup Capsule-Forensics model components"""
        try:
            # Get configuration
            num_classes = getattr(self.config.model, "num_classes", 2)
            config_gpu_id = getattr(self.config.model, "model_args", {}).get(
                "gpu_id", 0
            )

            # Determine actual GPU ID based on device availability
            if (
                self.device.type == "cuda"
                and torch.cuda.is_available()
                and config_gpu_id >= 0
            ):
                gpu_id = config_gpu_id
                print(f"   🔥 Using GPU {gpu_id}")
            else:
                gpu_id = -1  # Use -1 to indicate CPU mode
                if config_gpu_id >= 0:
                    print(
                        f"   ⚠️  GPU {config_gpu_id} requested but not available, falling back to CPU"
                    )
                else:
                    print(f"   💻 Using CPU mode as configured")

            # Initialize model components
            self.vgg_ext = self.model_big.VggExtractor()
            self.capnet = self.model_big.CapsuleNet(num_classes, gpu_id)
            self.capsule_loss = self.model_big.CapsuleLoss(gpu_id)

            # Setup optimizer
            lr = getattr(self.config.training, "learning_rate", 0.0005)
            beta1 = getattr(self.config.training, "optimizer_params", {}).get(
                "beta1", 0.9
            )
            self.optimizer = Adam(self.capnet.parameters(), lr=lr, betas=(beta1, 0.999))

            # Handle resuming from checkpoint
            resume_epoch = self.params.get("resume_epoch", 0)
            if resume_epoch > 0:
                self._load_checkpoint(resume_epoch)

            # Move to device based on gpu_id
            if gpu_id >= 0 and self.device.type == "cuda" and torch.cuda.is_available():
                self.capnet.cuda(gpu_id)
                self.vgg_ext.cuda(gpu_id)
                self.capsule_loss.cuda(gpu_id)
                print(f"   📱 Models moved to GPU {gpu_id}")
            else:
                # Ensure models are on CPU
                self.capnet.cpu()
                self.vgg_ext.cpu()
                self.capsule_loss.cpu()
                print(f"   💻 Models running on CPU")

            print(f"   ✅ Capsule-Forensics models initialized")
            print(f"      CapsuleNet: {num_classes} classes")
            print(f"      VGG Extractor: Ready")
            print(f"      CapsuleLoss: Ready")
            print(f"      Optimizer: Adam (lr={lr}, beta1={beta1})")

        except Exception as e:
            raise RuntimeError(f"Failed to setup Capsule-Forensics models: {e}")

    def _load_checkpoint(self, epoch: int):
        """Load checkpoint from specific epoch"""
        try:
            # Look for checkpoint in output directory or previous runs
            checkpoint_paths = [
                self.output_dir / f"capsule_{epoch}.pt",
                Path(f"checkpoints/binary_faceforensicspp/capsule_{epoch}.pt"),
                Path(f"training_output/capsule_forensics/capsule_{epoch}.pt"),
            ]

            checkpoint_path = None
            for path in checkpoint_paths:
                if path.exists():
                    checkpoint_path = path
                    break

            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")

            # Load model state
            self.capnet.load_state_dict(torch.load(checkpoint_path))
            self.capnet.train(mode=True)

            # Try to load optimizer state
            optim_path = checkpoint_path.parent / f"optim_{epoch}.pt"
            if optim_path.exists():
                self.optimizer.load_state_dict(torch.load(optim_path))

                # Move optimizer state to GPU if needed
                if self.device.type == "cuda":
                    gpu_id = (
                        int(str(self.device).split(":")[-1])
                        if ":" in str(self.device)
                        else 0
                    )
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(gpu_id)

                print(f"   ✅ Loaded checkpoint and optimizer from epoch {epoch}")
            else:
                print(
                    f"   ✅ Loaded checkpoint from epoch {epoch} (optimizer state not found)"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from epoch {epoch}: {e}")

    def train(self) -> Dict[str, Any]:
        """Main training loop using original Capsule-Forensics methodology"""
        print("🚀 Starting Capsule-Forensics Training")
        print("=" * 60)

        try:
            # Training parameters
            epochs = getattr(self.config.training, "epochs", 25)
            start_epoch = self.params.get("resume_epoch", 0)
            dropout = getattr(self.config.model, "dropout", 0.05)
            random_routing = not getattr(self.config.model, "model_args", {}).get(
                "disable_random", False
            )

            # Setup CSV logging (original behavior)
            self._setup_logging()

            print(f"📋 Training Configuration:")
            print(f"   Epochs: {epochs} (starting from {start_epoch})")
            print(f"   Dropout: {dropout}")
            print(f"   Random routing: {random_routing}")
            print(f"   Batch size: {self.config.data.batch_size}")
            print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")

            # Training loop
            for epoch in range(start_epoch, epochs):
                print(f"\n📅 Epoch {epoch+1}/{epochs}")

                # Training phase
                train_loss = self._train_epoch(epoch, dropout, random_routing)

                # Validation phase
                val_acc = self._validate_epoch(epoch)

                # Save checkpoint (original format)
                self._save_checkpoint(epoch)

                # Update best accuracy
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    # Save best model
                    self._save_best_checkpoint(epoch)

                # Log to CSV (original behavior)
                self._log_epoch(epoch, train_loss, val_acc)

                # Store for train_system compatibility
                self.train_losses.append(train_loss)
                self.val_accuracies.append(val_acc)

                print(f"   Train Loss: {train_loss:.6f}")
                print(f"   Val Accuracy: {val_acc:.4f}%")
                print(f"   Best Accuracy: {self.best_acc:.4f}%")

            # Create final results
            results = self._create_results()

            # Save train_system compatible checkpoint
            self._save_train_system_checkpoint(results)

            print("✅ Capsule-Forensics training completed!")
            print(f"   Best accuracy: {self.best_acc:.4f}%")
            print(f"   Results saved to: {self.output_dir}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise e

    def _train_epoch(self, epoch: int, dropout: float, random_routing: bool) -> float:
        """Train one epoch using original methodology"""
        self.capnet.train()
        self.vgg_ext.eval()  # VGG always in eval mode

        total_loss = 0.0
        num_batches = 0

        with tqdm(self.train_loader, desc=f"Training") as pbar:
            for i, (images, labels) in enumerate(pbar):
                # Move to device
                if self.device.type == "cuda":
                    gpu_id = (
                        int(str(self.device).split(":")[-1])
                        if ":" in str(self.device)
                        else 0
                    )
                    images = images.cuda(gpu_id)
                    labels = labels.cuda(gpu_id)

                # Convert to Variables (original code style)
                images = Variable(images)
                labels = Variable(labels)

                # Zero gradients
                self.optimizer.zero_grad()

                # Extract VGG features
                with torch.no_grad():
                    vgg_features = self.vgg_ext(images)

                # Forward pass through CapsuleNet
                classes, class_probs = self.capnet(
                    vgg_features, random_routing, dropout
                )

                # Compute loss
                loss = self.capsule_loss(classes, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        return total_loss / num_batches

    def _validate_epoch(self, epoch: int) -> float:
        """Validate using original methodology"""
        self.capnet.eval()
        self.vgg_ext.eval()

        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                if self.device.type == "cuda":
                    gpu_id = (
                        int(str(self.device).split(":")[-1])
                        if ":" in str(self.device)
                        else 0
                    )
                    images = images.cuda(gpu_id)
                    labels = labels.cuda(gpu_id)

                # Extract VGG features
                vgg_features = self.vgg_ext(images)

                # Forward pass
                classes, class_probs = self.capnet(vgg_features, random=False)

                # Get predictions using original methodology
                output_probs = class_probs.data.cpu().numpy()
                predicted = np.zeros(output_probs.shape[0], dtype=np.float32)

                for i in range(output_probs.shape[0]):
                    if output_probs[i, 1] >= output_probs[i, 0]:
                        predicted[i] = 1.0
                    else:
                        predicted[i] = 0.0

                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum()

                # Store for detailed metrics
                all_predictions.extend(predicted)
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * correct / total
        return accuracy

    def _setup_logging(self):
        """Setup CSV logging (original behavior)"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resume_epoch = self.params.get("resume_epoch", 0)
        if resume_epoch > 0:
            self.csv_writer = open(self.output_dir / "train.csv", "a")
        else:
            self.csv_writer = open(self.output_dir / "train.csv", "w")
            self.csv_writer.write("epoch,train_loss,val_accuracy\n")

    def _log_epoch(self, epoch: int, train_loss: float, val_acc: float):
        """Log epoch results to CSV"""
        self.csv_writer.write(f"{epoch},{train_loss:.6f},{val_acc:.4f}\n")
        self.csv_writer.flush()

    def _save_checkpoint(self, epoch: int):
        """Save checkpoint in original format"""
        # Save model
        torch.save(self.capnet.state_dict(), self.output_dir / f"capsule_{epoch}.pt")

        # Save optimizer
        torch.save(self.optimizer.state_dict(), self.output_dir / f"optim_{epoch}.pt")

    def _save_best_checkpoint(self, epoch: int):
        """Save best checkpoint"""
        # Copy current checkpoint as best
        shutil.copy2(
            self.output_dir / f"capsule_{epoch}.pt", self.output_dir / "capsule_best.pt"
        )

    def _create_results(self) -> Dict[str, Any]:
        """Create training results dictionary"""
        return {
            "final_accuracy": self.best_acc,
            "best_epoch": len(self.val_accuracies) - 1,  # Approximate
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "training_method": "capsule_forensics_original",
        }

    def _save_train_system_checkpoint(self, results: Dict[str, Any]):
        """Save train_system compatible checkpoint"""
        try:
            checkpoint = {
                "epoch": results["best_epoch"],
                "model_state_dict": self.capnet.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_acc,
                "train_losses": self.train_losses,
                "val_accuracies": self.val_accuracies,
                "config": (
                    self.config.to_dict()
                    if hasattr(self.config, "to_dict")
                    else vars(self.config)
                ),
                "training_method": "capsule_forensics_original",
            }

            # Save train_system format
            torch.save(checkpoint, self.output_dir / "best_checkpoint.pth")
            torch.save(checkpoint, self.output_dir / "last_checkpoint.pth")

            print(f"   💾 Saved train_system checkpoint: best_checkpoint.pth")

        except Exception as e:
            print(f"⚠️  Warning: Could not save train_system checkpoint: {e}")

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "csv_writer") and self.csv_writer:
            self.csv_writer.close()


if __name__ == "__main__":
    print("📄 Capsule-Forensics External Trainer loaded")
    print("   Available trainers:")
    print(
        "   - CapsuleForensicsTrainer: Original Capsule-Forensics-v2 training methodology"
    )
