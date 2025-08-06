#!/usr/bin/env python3
"""
Basic Example: Training a ResNet18 model using Train System

This example shows how to:
1. Create a simple training configuration
2. Train a model using the configuration
3. Monitor the training progress
"""

import torch
import torchvision.models as models
from train_system import UnifiedTrainingWrapper, ModelFactory
from train_system.config import UnifiedTrainingConfig, ConfigTemplateManager
from train_system.core import UnifiedTrainer


def example_direct_usage():
    """Example of direct model wrapping without configuration files"""
    print("=== Direct Usage Example ===")

    # Create a model
    model = models.resnet18(pretrained=True)

    # Wrap it with our training wrapper
    wrapper = ModelFactory.create_wrapped_model(model, num_classes=2)

    # The wrapper can now be used for training like any PyTorch model
    print(f"Model wrapped successfully!")
    print(f"Parameters: {wrapper.get_parameter_count():,}")

    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224)
    output = wrapper(test_input)
    print(f"Output shape: {output.shape}")

    # Get prediction dictionary for inference
    pred_dict = wrapper.predict(test_input)
    print(f"Prediction keys: {list(pred_dict.keys())}")


def example_config_based_training():
    """Example of configuration-based training"""
    print("\n=== Configuration-Based Training Example ===")

    # Create configuration from template
    config = ConfigTemplateManager.create_config_from_template(
        "torchvision",
        model={"architecture": "resnet18", "num_classes": 2},
        data={
            "train_path": "data/train",  # Replace with your data path
            "val_path": "data/val",  # Replace with your data path
            "batch_size": 16,
        },
        training={"epochs": 5, "learning_rate": 0.001},
        output={"experiment_name": "example_experiment"},
    )

    print("Configuration created successfully!")
    print(f"Model: {config.model.name}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Experiment: {config.output.experiment_name}")

    # Note: Uncomment below to actually train (requires data)
    # trainer = UnifiedTrainer(config)
    # results = trainer.train()
    # print(f"Training completed! Best accuracy: {results['best_val_accuracy']:.2f}%")


def example_model_factory():
    """Example of using ModelFactory utilities"""
    print("\n=== Model Factory Example ===")

    # Create from torchvision
    wrapper1 = ModelFactory.create_wrapped_model(
        models.resnet18(pretrained=True), num_classes=2
    )
    print(f"ResNet18 wrapper created: {wrapper1.get_parameter_count():,} parameters")

    # Using model utilities
    from train_system.core.wrapper import ModelUtils

    # List available adapters
    adapters = ModelUtils.list_available_adapters()
    print(f"Available adapters: {list(adapters.keys())}")

    # Test model compatibility
    model = models.efficientnet_b0(pretrained=True)
    compatibility = ModelUtils.validate_model_compatibility(model)
    print(f"EfficientNet-B0 compatibility: {compatibility}")


def example_configuration_templates():
    """Example of working with configuration templates"""
    print("\n=== Configuration Templates Example ===")

    # List available templates
    templates = ["torchvision", "generic", "blip"]
    print(f"Available templates: {templates}")

    # Get a template
    template = ConfigTemplateManager.get_template("torchvision")
    print("Torchvision template structure:")
    for key in template.keys():
        print(f"  {key}: {type(template[key])}")

    # Save template to file
    ConfigTemplateManager.save_template("torchvision", "example_config.yaml")
    print("Template saved to example_config.yaml")


if __name__ == "__main__":
    print("Train System Basic Examples")
    print("=" * 50)

    try:
        example_direct_usage()
        example_config_based_training()
        example_model_factory()
        example_configuration_templates()

        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your dataset in the required format")
        print("2. Create a configuration file using: train-system create-template")
        print("3. Train your model using: train-system train config.yaml")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
