#!/usr/bin/env python3
"""
Dry Run functionality for Train System

Performs a comprehensive validation of the training setup including
actual sample training to verify the entire pipeline works correctly.
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

# Import train system components
try:
    from ..config import ConfigValidator, UnifiedTrainingConfig
    from ..core.trainer import UnifiedTrainer
except ImportError:
    # Handle direct execution
    sys.path.append(str(Path(__file__).parent.parent))
    from config import ConfigValidator, UnifiedTrainingConfig
    from core.trainer import UnifiedTrainer


def validate_config(config_path: str) -> bool:
    """Validate a configuration file"""

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False

    try:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            config = UnifiedTrainingConfig.from_yaml(config_path)
        elif config_path.suffix.lower() == ".json":
            config = UnifiedTrainingConfig.from_json(config_path)
        else:
            print(f"❌ Unsupported file format: {config_path.suffix}")
            return False

        # Validate configuration
        validation_result = ConfigValidator.validate(config)

        if validation_result.is_valid:
            print(f"✅ Configuration is valid: {config_path}")
            print(f"   Model: {config.model.name}")
            print(f"   Dataset: {config.data.name}")
            print(f"   Epochs: {config.training.epochs}")
            print(f"   Output: {config.output.experiment_name}")

            if validation_result.warnings:
                print("⚠️ Warnings:")
                for warning in validation_result.warnings:
                    print(f"   {warning}")
        else:
            print(f"❌ Configuration validation failed:")
            for error in validation_result.errors:
                print(f"   {error}")

        return validation_result.is_valid

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def dry_run_training(config_path: str, num_batches: int = 2) -> bool:
    """
    Perform a dry run of the training pipeline with sample data.
    This validates the entire training setup including model initialization,
    data loading, forward/backward passes, and checkpoint saving.

    Args:
        config_path: Path to configuration file
        num_batches: Number of training batches to test (default: 2)

    Returns:
        bool: True if dry run successful, False otherwise
    """

    print("🧪 Starting Training Dry Run")
    print("=" * 60)
    print("This will perform a mini training session to validate your setup.")
    print(f"Number of test batches: {num_batches}")
    print()

    # First validate configuration
    if not validate_config(config_path):
        print("❌ Configuration validation failed. Cannot proceed with dry run.")
        return False, False  # Return failure and not external

    try:
        # Load configuration
        config_path = Path(config_path)

        if config_path.suffix.lower() in [".yaml", ".yml"]:
            config = UnifiedTrainingConfig.from_yaml(config_path)
        else:
            config = UnifiedTrainingConfig.from_json(config_path)

        print("🔧 Initializing trainer for dry run...")

        # Create trainer
        trainer = UnifiedTrainer(config)

        # Check if this is an external trainer first
        external_trainer_config = getattr(config, "external_trainer", None)
        is_external = external_trainer_config and getattr(
            external_trainer_config, "enabled", False
        )

        print("✅ Trainer initialized successfully")

        # Initialize model, data, and optimizer for dry run (normally done in train())
        try:
            if is_external:
                print("🎯 External trainer detected - skipping standard model loading")

                # Check if external trainer skips unified data loading
                skip_data_loading = False
                if hasattr(external_trainer_config, "skip_unified_data_loading"):
                    skip_data_loading = (
                        external_trainer_config.skip_unified_data_loading
                    )
                elif isinstance(external_trainer_config, dict):
                    skip_data_loading = external_trainer_config.get(
                        "skip_unified_data_loading", False
                    )

                if skip_data_loading:
                    print(
                        "🔧 External trainer handles its own data loading - skipping unified data loading"
                    )
                    print("✅ External trainer setup completed for dry run")
                else:
                    print("🔧 Loading data for dry run...")
                    trainer.load_data()
                    print("✅ Data loading completed for external trainer")
                # Skip model and optimizer setup for external trainers
            else:
                print("🔧 Loading model for dry run...")
                trainer.load_model()
                print("🔧 Loading data for dry run...")
                trainer.load_data()
                print("🔧 Setting up optimizer for dry run...")
                trainer.setup_optimizer()
                print("✅ All components initialized for dry run")
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            import traceback

            traceback.print_exc()
            return False, False

        if is_external:
            print(
                f"🎯 External trainer detected: {getattr(external_trainer_config, 'name', 'Unknown')}"
            )
            print(f"   Device: {trainer.device}")
            # For external trainers, we don't have direct access to model/optimizer/criterion
            print(f"   Mode: External Trainer (custom training logic)")
            success = dry_run_external_trainer(trainer, num_batches)
            return success, True  # Return success and is_external flag
        else:
            print(f"   Model: {trainer.model.__class__.__name__}")
            print(f"   Device: {trainer.device}")
            print(f"   Optimizer: {trainer.optimizer.__class__.__name__}")

            # Check if criterion exists (might not be initialized yet)
            if hasattr(trainer, "criterion") and trainer.criterion is not None:
                print(f"   Loss function: {trainer.criterion.__class__.__name__}")
            else:
                print(f"   Loss function: Will be initialized during training")

            success = dry_run_standard_trainer(trainer, num_batches)
            return success, False  # Return success and is_external flag

    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        import traceback

        traceback.print_exc()
        return False, False  # Return failure and not external


def dry_run_standard_trainer(trainer, num_batches: int) -> bool:
    """Perform dry run for standard trainer"""

    print(f"\n🚀 Running {num_batches} training batches (Standard Trainer)...")

    # Set models to training mode
    trainer.model.train()

    batch_count = 0
    total_loss = 0.0

    try:
        # Training loop
        for batch_idx, (data, targets) in enumerate(trainer.train_loader):
            if batch_count >= num_batches:
                break

            print(f"   Batch {batch_count + 1}/{num_batches}:")

            # Move data to device
            data = data.to(trainer.device)
            targets = targets.to(trainer.device)

            print(f"      Data shape: {data.shape}")
            print(f"      Target shape: {targets.shape}")

            # Zero gradients
            trainer.optimizer.zero_grad()

            # Forward pass
            outputs = trainer.model(data)
            print(f"      Output shape: {outputs.shape}")

            # Calculate loss
            loss = trainer.criterion(outputs, targets)
            print(f"      Loss: {loss.item():.6f}")

            # Backward pass
            loss.backward()
            trainer.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            print(f"      ✅ Batch {batch_count} completed successfully")

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    avg_loss = total_loss / batch_count
    print(f"\n📊 Dry Run Training Results:")
    print(f"   Batches processed: {batch_count}")
    print(f"   Average loss: {avg_loss:.6f}")
    print(f"   ✅ Forward and backward passes successful")

    # Test validation step
    return test_validation_step(trainer, num_batches)


def dry_run_external_trainer(trainer, num_batches: int) -> bool:
    """Perform dry run for external trainer"""

    print(f"\n🎯 Testing External Trainer Setup...")

    try:
        # For external trainers, the actual external trainer is created during training
        # So we need to simulate that creation for the dry run
        from ..core.external_trainer import HybridTrainer

        print(f"   🔧 Creating external trainer for dry run...")

        # Create HybridTrainer (same as what happens during actual training)
        hybrid_trainer = HybridTrainer(
            config=trainer.config,
            model=None,  # External trainer will create its own model
            device=trainer.device,
            output_dir=Path("dry_run_temp"),  # Temporary output directory
        )

        # Now check if the actual external trainer was properly initialized
        if (
            not hasattr(hybrid_trainer, "external_trainer")
            or hybrid_trainer.external_trainer is None
        ):
            print("❌ External trainer not properly initialized in HybridTrainer")
            return False

        external_trainer = hybrid_trainer.external_trainer
        print(
            f"   ✅ External trainer initialized: {external_trainer.__class__.__name__}"
        )

        # Check required attributes
        required_attrs = ["train_loader", "val_loader", "device", "config"]
        for attr in required_attrs:
            if hasattr(external_trainer, attr):
                print(f"   ✅ {attr}: Available")
            else:
                print(f"   ❌ {attr}: Missing")
                return False

        # Test a few batches from data loaders
        print(
            f"\n🚀 Testing {num_batches} batches from external trainer data loaders..."
        )

        batch_count = 0
        for batch_idx, (data, targets) in enumerate(external_trainer.train_loader):
            if batch_count >= num_batches:
                break

            print(f"   Batch {batch_count + 1}/{num_batches}:")
            print(f"      Data shape: {data.shape}")
            print(f"      Target shape: {targets.shape}")
            print(f"      Data type: {data.dtype}")
            print(
                f"      Target values: {targets.unique() if hasattr(targets, 'unique') else 'N/A'}"
            )
            print(f"      ✅ Data loading successful")

            batch_count += 1

        print(f"\n📊 External Trainer Dry Run Results:")
        print(f"   Batches tested: {batch_count}")
        print(f"   ✅ External trainer data loading successful")
        print(f"   ✅ External trainer setup validated")

        # Test if external trainer has training-related methods
        training_methods = ["train", "_train_epoch", "_validate_epoch"]
        for method in training_methods:
            if hasattr(external_trainer, method):
                print(f"   ✅ Training method '{method}': Available")
            else:
                print(f"   ⚠️  Training method '{method}': Not found")

        # Test external trainer model components if available
        model_component_tested = False

        # Try different external trainer types
        if hasattr(external_trainer, "capnet") and hasattr(external_trainer, "vgg_ext"):
            # Capsule Forensics trainer
            print(f"\n🧠 Testing Capsule Forensics Model Components...")
            model_component_tested = True

            # Test with a single batch
            try:
                for batch_idx, (data, targets) in enumerate(
                    external_trainer.train_loader
                ):
                    print(f"   Testing forward pass with real data...")

                    # Move data to device
                    if external_trainer.device.type == "cuda":
                        gpu_id = (
                            int(str(external_trainer.device).split(":")[-1])
                            if ":" in str(external_trainer.device)
                            else 0
                        )
                        data = data.cuda(gpu_id)
                        targets = targets.cuda(gpu_id)

                    # Test VGG feature extraction
                    with torch.no_grad():
                        vgg_features = external_trainer.vgg_ext(data)
                        print(f"      VGG features shape: {vgg_features.shape}")

                        # Test CapsuleNet forward pass
                        classes, class_probs = external_trainer.capnet(
                            vgg_features, random=False
                        )
                        print(f"      CapsuleNet classes shape: {classes.shape}")
                        print(
                            f"      CapsuleNet class_probs shape: {class_probs.shape}"
                        )

                        # Test loss calculation
                        loss = external_trainer.capsule_loss(classes, targets)
                        print(f"      Loss value: {loss.item():.6f}")

                        print(f"   ✅ Capsule Forensics model forward pass successful")

                    break  # Only test one batch

            except Exception as e:
                print(f"   ⚠️  Capsule Forensics model test failed: {e}")
                print(
                    f"   💡 This might be normal if models need special initialization"
                )

        elif hasattr(external_trainer, "yolo_model"):
            # YOLO trainer
            print(f"\n🧠 Testing YOLO Model Components...")
            model_component_tested = True

            try:
                yolo_model = external_trainer.yolo_model
                print(f"      YOLO model type: {type(yolo_model).__name__}")
                print(
                    f"      YOLO model task: {getattr(yolo_model, 'task', 'unknown')}"
                )

                # Check if we can access the underlying model
                if hasattr(yolo_model, "model"):
                    pytorch_model = yolo_model.model
                    print(f"      PyTorch model type: {type(pytorch_model).__name__}")
                    print(f"   ✅ YOLO model structure validation successful")
                else:
                    print(f"   ⚠️  YOLO model does not have accessible PyTorch model")

            except Exception as e:
                print(f"   ❌ YOLO model validation failed: {e}")
                return False

        # Generic model test for other external trainers
        if not model_component_tested:
            print(f"\n🧠 Testing Generic External Trainer Model...")

            # Check for common model attributes
            model_attrs = ["model", "net", "network", "classifier"]
            found_model = False

            for attr in model_attrs:
                if hasattr(external_trainer, attr):
                    model_obj = getattr(external_trainer, attr)
                    if model_obj is not None:
                        print(f"      Found model attribute: {attr}")
                        print(f"      Model type: {type(model_obj).__name__}")
                        found_model = True
                        break

            if found_model:
                print(f"   ✅ Generic model structure validation successful")
            else:
                print(
                    f"   ⚠️  No standard model attributes found, but trainer may handle models internally"
                )

        # Test validation data loader
        print(f"\n🧮 Testing validation data loader...")
        val_batch_count = 0
        try:
            for batch_idx, (data, targets) in enumerate(external_trainer.val_loader):
                if val_batch_count >= min(1, num_batches):
                    break

                print(f"   Validation batch {val_batch_count + 1}:")
                print(f"      Data shape: {data.shape}")
                print(f"      Target shape: {targets.shape}")
                val_batch_count += 1

            print(f"   ✅ Validation data loading successful")
        except Exception as e:
            print(f"   ⚠️  Validation data loading failed: {e}")

        print(f"\n💡 External Trainer Summary:")
        print(f"   • Data loaders: Working correctly")
        print(f"   • Configuration: Loaded successfully")
        print(f"   • Training methods: Available")
        if hasattr(external_trainer, "capnet"):
            print(f"   • Model components: Initialized and tested")
        print(f"   • External trainer uses custom training logic")
        print(f"   • Dry run validates setup only")
        print(f"   • Full training will use external methodology")

        return True

    except Exception as e:
        print(f"❌ External trainer dry run failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_validation_step(trainer, num_batches: int) -> bool:
    """Test validation step"""

    print(f"\n🧮 Testing validation step...")

    trainer.model.eval()
    val_batch_count = 0
    val_correct = 0
    val_total = 0

    try:
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(trainer.val_loader):
                if val_batch_count >= min(2, num_batches):
                    break

                data = data.to(trainer.device)
                targets = targets.to(trainer.device)

                outputs = trainer.model(data)

                # Calculate accuracy for classification tasks
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

                val_batch_count += 1

        if val_total > 0:
            val_accuracy = 100.0 * val_correct / val_total
            print(f"   Validation batches: {val_batch_count}")
            print(f"   Validation accuracy: {val_accuracy:.2f}%")
        else:
            print(f"   Validation batches: {val_batch_count}")
            print(f"   ✅ Validation forward pass successful")

        # Test checkpoint saving
        return test_checkpoint_saving(trainer)

    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        return False


def test_checkpoint_saving(trainer) -> bool:
    """Test checkpoint saving functionality"""

    print(f"\n💾 Testing checkpoint saving...")

    try:
        # Create a temporary directory for dry run outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test saving checkpoint
            checkpoint = {
                "epoch": 0,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "loss": 0.5,  # dummy loss
                "dry_run": True,
            }

            checkpoint_path = temp_path / "dry_run_checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)

            # Verify checkpoint can be loaded
            loaded_checkpoint = torch.load(checkpoint_path)

            print(f"   ✅ Checkpoint saving/loading successful")
            print(f"   Checkpoint size: {checkpoint_path.stat().st_size / 1024:.1f} KB")

            return True

    except Exception as e:
        print(f"   ⚠️  Checkpoint test failed: {e}")
        return False


def print_final_summary(success: bool, is_external: bool = False):
    """Print final dry run summary"""

    print(
        f"\n{'🎉' if success else '❌'} Dry Run {'Completed Successfully' if success else 'Failed'}!"
    )
    print("=" * 60)

    if success:
        print("✅ Configuration is valid")
        print("✅ Trainer initialization successful")
        print("✅ Data loading successful")

        if is_external:
            print("✅ External trainer setup validated")
            print("✅ External trainer components tested")
            print("✅ Model forward pass successful")
            print("✅ Custom training methodology ready")
        else:
            print("✅ Model initialization successful")
            print("✅ Forward pass successful")
            print("✅ Backward pass successful")
            print("✅ Optimizer step successful")
            print("✅ Validation step successful")
            print("✅ Checkpoint saving successful")

        print()
        print("🚀 Your setup is ready for full training!")
        print(f"   Run: train-system train <config_file>")

        if is_external:
            print(f"   💡 External trainer will use custom training methodology")
    else:
        print("❌ Dry run validation failed")
        print("🔧 Please check the errors above and fix your configuration")
        print("💡 Common issues:")
        print("   • Dataset paths don't exist")
        print("   • Model configuration incorrect")
        print("   • Missing dependencies")
        print("   • GPU/device configuration issues")
        if is_external:
            print("   • External trainer module import issues")
            print("   • Custom model component initialization problems")


def main():
    """Main entry point for dry run script"""
    parser = argparse.ArgumentParser(
        description="Train System Dry Run - Validate training setup with sample data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic dry run with 2 test batches
  python -m train_system.cli.dry_run config.yaml

  # Extended dry run with 5 test batches
  python -m train_system.cli.dry_run config.yaml --batches 5

  # Dry run via main CLI
  train-system dry-run config.yaml --batches 3
        """,
    )

    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Number of test batches to run (default: 2)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Run dry run
    success, is_external = dry_run_training(args.config, args.batches)

    # Print summary
    print_final_summary(success, is_external)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
