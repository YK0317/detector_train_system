#!/usr/bin/env python3
"""
Unified Training Wrapper Core

The main wrapper class that makes any standalone model trainable.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import os
import importlib.util

from ..adapters import ModelAdapter, AutoAdapter


class UnifiedTrainingWrapper(nn.Module):
    """
    Universal training wrapper that can make any standalone model trainable
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: Optional[ModelAdapter] = None,
        num_classes: int = 2,
        loss_function: Optional[nn.Module] = None,
    ):
        """
        Initialize the unified training wrapper

        Args:
            model: The standalone model to wrap
            adapter: Model adapter for handling different output formats
            num_classes: Number of classes for classification
            loss_function: Loss function to use (default: CrossEntropyLoss)
        """
        super().__init__()

        self.model = model
        self.num_classes = num_classes

        # Auto-detect adapter if not provided
        if adapter is None:
            self.adapter = AutoAdapter()
            logging.info("Using AutoAdapter for automatic output format detection")
        else:
            self.adapter = adapter
            logging.info(f"Using {type(adapter).__name__} for output format handling")

        # Set loss function
        if loss_function is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training-compatible forward pass
        Always returns only logits for training
        """
        model_output = self.model(x)
        logits = self.adapter.extract_logits(model_output)
        return logits

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference method that returns full prediction dictionary
        """
        self.eval()
        with torch.no_grad():
            model_output = self.model(x)
            return self.adapter.get_prediction_dict(model_output)

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for training"""
        return self.loss_fn(outputs, targets)

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_weights(self, filepath: str):
        """Save model weights"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "adapter_type": type(self.adapter).__name__,
            },
            filepath,
        )
        logging.info(f"Model weights saved to: {filepath}")

    def load_weights(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Model weights loaded from: {filepath}")


class ModelFactory:
    """
    Generic factory for creating wrapped models without hardcoded imports.
    Supports dynamic model loading and configuration-based setup.
    """

    @staticmethod
    def create_wrapped_model(
        model: nn.Module,
        adapter: Optional[ModelAdapter] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> UnifiedTrainingWrapper:
        """
        Create a wrapped model from an existing PyTorch model instance

        Args:
            model: Any PyTorch model instance
            adapter: Optional adapter (AutoAdapter used if None)
            model_path: Optional path to pretrained weights
            **kwargs: Additional arguments for the wrapper

        Returns:
            UnifiedTrainingWrapper instance

        Example:
            # Method 1: With any PyTorch model
            my_model = MyCustomModel()
            wrapper = ModelFactory.create_wrapped_model(my_model)

            # Method 2: With specific adapter
            wrapper = ModelFactory.create_wrapped_model(my_model, StandardAdapter())

            # Method 3: Load pretrained weights
            wrapper = ModelFactory.create_wrapped_model(my_model, model_path='weights.pth')
        """

        # Use AutoAdapter if no adapter specified
        if adapter is None:
            adapter = AutoAdapter()
            logging.info(
                "No adapter specified, using AutoAdapter for automatic detection"
            )

        # Load pretrained weights if provided
        if model_path:
            ModelFactory._load_weights(model, model_path)

        # Create wrapper
        wrapper = UnifiedTrainingWrapper(model, adapter, **kwargs)
        logging.info(
            f"Created wrapper for {type(model).__name__} with {type(adapter).__name__}"
        )
        return wrapper

    @staticmethod
    def from_config(config: Dict[str, Any]) -> UnifiedTrainingWrapper:
        """
        Create a wrapped model from configuration dictionary

        Args:
            config: Configuration dictionary with model setup

        Example config:
            {
                "model_class": "torchvision.models.resnet18",
                "model_args": {"num_classes": 2},
                "adapter": "AutoAdapter",
                "model_path": "path/to/weights.pth",
                "num_classes": 2
            }
        """

        # Import model class dynamically
        model_class_path = config.get("model_class")
        if not model_class_path:
            raise ValueError("config must contain 'model_class'")

        model_class = ModelFactory._import_class(model_class_path)
        model_args = config.get("model_args", {})
        model = model_class(**model_args)

        # Get adapter with optional configuration
        adapter = None

        # Check for external_adapter first (alternative method)
        if hasattr(config, "external_adapter") and config.external_adapter:
            external_adapter_config = config.external_adapter
            from ..adapters.external_adapter import ExternalAdapterLoader

            try:
                # Use the ExternalAdapterLoader for compatibility
                adapter = ExternalAdapterLoader.get_adapter(config)
            except Exception as e:
                logging.error(f"Failed to load external adapter: {e}")
                # Fall back to default adapter
                adapter_name = "auto"
                adapter_config = {}
                adapter = ModelFactory._get_adapter(adapter_name, adapter_config)
        else:
            # Use standard adapter configuration
            adapter_name = config.get("adapter", "auto")
            adapter_config = config.get("adapter_config", {})
            adapter = ModelFactory._get_adapter(adapter_name, adapter_config)

        # Other wrapper args
        wrapper_args = {
            k: v
            for k, v in config.items()
            if k
            not in [
                "model_class",
                "model_args",
                "adapter",
                "adapter_config",
                "model_path",
            ]
        }

        return ModelFactory.create_wrapped_model(
            model=model,
            adapter=adapter,
            model_path=config.get("model_path"),
            **wrapper_args,
        )

    @staticmethod
    def _import_class(class_path: str):
        """Dynamically import a class from string path"""
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    @staticmethod
    def _get_adapter(
        adapter_name: str, adapter_config: Dict[str, Any] = None
    ) -> ModelAdapter:
        """Get adapter instance by name with optional configuration"""
        from ..adapters import (
            AutoAdapter,
            StandardAdapter,
            LogitsAndFeaturesAdapter,
            DictOutputAdapter,
            UCFAdapter,
            CapsuleNetworkAdapter,
            get_adapter_for_model,
        )

        if adapter_config is None:
            adapter_config = {}

        # Try registry first for auto-discovery
        try:
            from ..registry import adapter_registry

            registry_adapter = adapter_registry.get_adapter(adapter_name)
            if registry_adapter:
                logging.info(f"Using registered adapter: {adapter_name}")
                # Extract adapter parameters (exclude registry-specific keys)
                adapter_params = {
                    k: v
                    for k, v in adapter_config.items()
                    if k not in ["script_path", "class_name", "required_packages"]
                }
                return registry_adapter(**adapter_params)
        except ImportError:
            logging.debug("Registry not available, using direct adapter mapping")
        except Exception as e:
            logging.debug(f"Registry lookup failed for '{adapter_name}': {e}")

        # Direct adapter class mapping (fallback)
        adapter_map = {
            "auto": AutoAdapter,
            "standard": StandardAdapter,
            "logits_features": LogitsAndFeaturesAdapter,
            "dict": DictOutputAdapter,
            "capsule": CapsuleNetworkAdapter,
            "ucf": UCFAdapter,
        }

        adapter_name_lower = adapter_name.lower()

        # Check for external adapter
        if adapter_name_lower == "external":
            from ..adapters.external_adapter import load_external_adapter

            if (
                "script_path" not in adapter_config
                or "class_name" not in adapter_config
            ):
                raise ValueError(
                    "External adapter requires 'script_path' and 'class_name' in adapter_config"
                )

            # Load the external adapter class
            script_path = adapter_config["script_path"]
            class_name = adapter_config["class_name"]

            # Remove these from adapter_config as they're not actual adapter parameters
            adapter_params = {
                k: v
                for k, v in adapter_config.items()
                if k not in ["script_path", "class_name"]
            }

            try:
                adapter_class = load_external_adapter(script_path, class_name)
                return adapter_class(**adapter_params)
            except Exception as e:
                logging.error(f"Failed to load external adapter: {e}")
                # Fall back to AutoAdapter in case of error
                return AutoAdapter()

        # Try direct class mapping first
        if adapter_name_lower in adapter_map:
            adapter_class = adapter_map[adapter_name_lower]
            try:
                return adapter_class(**adapter_config)
            except TypeError:
                # If adapter doesn't accept config parameters, create without them
                logging.warning(
                    f"Adapter {adapter_name} doesn't accept configuration parameters"
                )
                return adapter_class()

        # Try model-specific adapter lookup (for backwards compatibility)
        try:
            return get_adapter_for_model(adapter_name_lower)
        except Exception:
            pass

        # Try to import adapter class directly by name
        try:
            from ..adapters import model_adapters

            adapter_class = getattr(model_adapters, adapter_name)
            try:
                return adapter_class(**adapter_config)
            except TypeError:
                return adapter_class()
        except (AttributeError, ImportError):
            pass

        # Fall back to AutoAdapter
        logging.warning(f"Unknown adapter '{adapter_name}', using AutoAdapter")
        return AutoAdapter()

    @staticmethod
    def _load_weights(model: nn.Module, model_path: str):
        """Load weights into model"""
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location="cpu")

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Handle DataParallel keys
                if any(key.startswith("module.") for key in state_dict.keys()):
                    state_dict = {
                        key.replace("module.", ""): value
                        for key, value in state_dict.items()
                    }

                model.load_state_dict(state_dict, strict=False)
                logging.info(f"Loaded pretrained weights from: {model_path}")
            else:
                logging.warning(f"Model weights file not found: {model_path}")
        except Exception as e:
            logging.error(f"Could not load pretrained weights: {e}")
            raise


class ModelUtils:
    """
    Utility class for common model operations and helper functions
    """

    @staticmethod
    def create_from_torchvision(model_name: str, **kwargs) -> UnifiedTrainingWrapper:
        """
        Create wrapper from torchvision models

        Args:
            model_name: Name of torchvision model (e.g., 'resnet18', 'efficientnet_b4')
            **kwargs: Arguments for the model

        Example:
            wrapper = ModelUtils.create_from_torchvision('resnet18', num_classes=2)
        """
        import torchvision.models as models

        if not hasattr(models, model_name):
            raise ValueError(f"Model '{model_name}' not found in torchvision.models")

        model_class = getattr(models, model_name)
        model = model_class(**kwargs)

        return ModelFactory.create_wrapped_model(model)

    @staticmethod
    def create_from_timm(model_name: str, **kwargs) -> UnifiedTrainingWrapper:
        """
        Create wrapper from timm models

        Args:
            model_name: Name of timm model
            **kwargs: Arguments for the model

        Example:
            wrapper = ModelUtils.create_from_timm('efficientnet_b4', num_classes=2)
        """
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm library not installed. Install with: pip install timm"
            )

        model = timm.create_model(model_name, **kwargs)
        return ModelFactory.create_wrapped_model(model)

    @staticmethod
    def create_from_custom_path(
        model_file: str, class_name: str, **kwargs
    ) -> UnifiedTrainingWrapper:
        """
        Create wrapper from custom model file

        Args:
            model_file: Path to Python file containing model class
            class_name: Name of the model class
            **kwargs: Arguments for the model constructor

        Example:
            wrapper = ModelUtils.create_from_custom_path(
                'models/meso4_standalone.py',
                'Meso4',
                num_classes=2
            )
        """

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load module dynamically
        spec = importlib.util.spec_from_file_location("custom_model", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get model class
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {model_file}")

        model_class = getattr(module, class_name)
        model = model_class(**kwargs)

        return ModelFactory.create_wrapped_model(model)

    @staticmethod
    def list_available_adapters() -> Dict[str, str]:
        """List all available adapters with descriptions"""
        return {
            "AutoAdapter": "Automatically detects model output format (recommended)",
            "StandardAdapter": "For models returning only logits",
            "LogitsAndFeaturesAdapter": "For models returning (logits, features) tuple",
            "DictOutputAdapter": "For models returning dictionary outputs",
            "UCFAdapter": "Specialized for UCF models",
            "CapsuleNetworkAdapter": "Specialized for CapsuleNet models",
            "CustomFunctionAdapter": "Custom adapter with user-defined functions",
        }

    @staticmethod
    def validate_model_compatibility(
        model: nn.Module, test_input_shape: tuple = (1, 3, 224, 224)
    ) -> Dict[str, Any]:
        """
        Test model compatibility with the wrapper

        Args:
            model: PyTorch model to test
            test_input_shape: Shape of test input tensor

        Returns:
            Dictionary with compatibility information
        """
        test_input = torch.randn(*test_input_shape)

        try:
            model.eval()
            with torch.no_grad():
                output = model(test_input)

            # Analyze output
            output_info = {
                "compatible": True,
                "output_type": type(output).__name__,
                "output_shape": None,
                "recommended_adapter": "AutoAdapter",
            }

            if torch.is_tensor(output):
                output_info["output_shape"] = output.shape
                output_info["recommended_adapter"] = "StandardAdapter"
            elif isinstance(output, (tuple, list)):
                output_info["output_shape"] = [
                    item.shape if torch.is_tensor(item) else type(item).__name__
                    for item in output
                ]
                if len(output) == 2:
                    output_info["recommended_adapter"] = "LogitsAndFeaturesAdapter"
                elif len(output) == 3:
                    output_info["recommended_adapter"] = "CapsuleNetworkAdapter"
            elif isinstance(output, dict):
                output_info["output_keys"] = list(output.keys())
                output_info["recommended_adapter"] = "DictOutputAdapter"

            return output_info

        except Exception as e:
            return {
                "compatible": False,
                "error": str(e),
                "recommended_adapter": "AutoAdapter",
            }
