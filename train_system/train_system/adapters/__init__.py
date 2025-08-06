#!/usr/bin/env python3
"""
Model Adapters for Unified Training Wrapper

This module contains adapter classes that handle different model output formats.
Each adapter knows how to extract logits, features, and create prediction dictionaries
from specific model output formats.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    Each model type needs its own adapter to handle specific forward method signatures.
    """

    @abstractmethod
    def extract_logits(self, model_output: Any) -> torch.Tensor:
        """Extract logits from model output for training"""
        pass

    @abstractmethod
    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        """Extract features from model output for analysis (optional)"""
        pass

    @abstractmethod
    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """Convert model output to prediction dictionary for inference"""
        pass


class StandardAdapter(ModelAdapter):
    """Adapter for models that return only logits"""

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        return model_output

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        return None

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        logits = model_output
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }


class LogitsAndFeaturesAdapter(ModelAdapter):
    """Adapter for models that return (logits, features) tuple"""

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, (tuple, list)):
            return model_output[0]  # First element is logits
        return model_output

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        if isinstance(model_output, (tuple, list)) and len(model_output) > 1:
            return model_output[1]  # Second element is features
        return None

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        logits = self.extract_logits(model_output)
        features = self.extract_features(model_output)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }

        if features is not None:
            result["features"] = features

        return result


class DictOutputAdapter(ModelAdapter):
    """Adapter for models that return dictionary output"""

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, dict):
            # Try common keys for logits
            for key in ["logits", "outputs", "predictions", "scores"]:
                if key in model_output:
                    return model_output[key]
        return model_output

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        if isinstance(model_output, dict):
            for key in ["features", "embeddings", "hidden"]:
                if key in model_output:
                    return model_output[key]
        return None

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(model_output, dict):
            # If already a dict, add computed values
            result = model_output.copy()
            logits = self.extract_logits(model_output)
            result["probabilities"] = F.softmax(logits, dim=1)
            result["predictions"] = torch.argmax(logits, dim=1)
            return result
        else:
            # Convert to standard format
            return StandardAdapter().get_prediction_dict(model_output)


class UCFAdapter(ModelAdapter):
    """Adapter for UCF (Universal Consistency Forensics) models"""

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        """UCF models return logits directly"""
        if torch.is_tensor(model_output):
            return model_output
        else:
            raise ValueError(
                f"UCF model expected tensor output, got {type(model_output)}"
            )

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        """UCF doesn't expose features in this implementation"""
        return None

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        """Convert UCF output to prediction dictionary"""
        logits = self.extract_logits(model_output)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }


class CapsuleNetworkAdapter(ModelAdapter):
    """Adapter for CapsuleNet models that return multiple outputs"""

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, (tuple, list)):
            # CapsuleNet typically returns (classes, reconstructions, masked)
            # Classes are usually the first element
            return model_output[0]
        return model_output

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        if isinstance(model_output, (tuple, list)) and len(model_output) > 2:
            # Third element might be intermediate features
            return model_output[2]
        return None

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        logits = self.extract_logits(model_output)
        features = self.extract_features(model_output)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }

        if features is not None:
            result["capsule_features"] = features

        # If there are reconstructions (second element), include them
        if isinstance(model_output, (tuple, list)) and len(model_output) > 1:
            result["reconstructions"] = model_output[1]

        return result


class CustomFunctionAdapter(ModelAdapter):
    """Adapter that uses custom functions for extraction"""

    def __init__(
        self,
        logits_extractor: Callable[[Any], torch.Tensor],
        features_extractor: Optional[Callable[[Any], torch.Tensor]] = None,
        prediction_converter: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ):
        self.logits_extractor = logits_extractor
        self.features_extractor = features_extractor
        self.prediction_converter = prediction_converter

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        return self.logits_extractor(model_output)

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        if self.features_extractor:
            return self.features_extractor(model_output)
        return None

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        if self.prediction_converter:
            return self.prediction_converter(model_output)
        else:
            logits = self.extract_logits(model_output)
            return StandardAdapter().get_prediction_dict(logits)


class AutoAdapter(ModelAdapter):
    """Automatically detects the appropriate adapter based on model output"""

    def __init__(self):
        self.adapters = {
            "standard": StandardAdapter(),
            "tuple": LogitsAndFeaturesAdapter(),
            "dict": DictOutputAdapter(),
            "capsule": CapsuleNetworkAdapter(),
            "ucf": UCFAdapter(),
        }
        self.detected_type = None

    def _detect_output_type(self, model_output: Any) -> str:
        """Detect the output type and cache it"""
        if isinstance(model_output, dict):
            return "dict"
        elif isinstance(model_output, (tuple, list)):
            # Check if it's a capsule network output (3 elements)
            if len(model_output) == 3:
                return "capsule"
            else:
                return "tuple"
        else:
            return "standard"

    def _get_adapter(self, model_output: Any) -> ModelAdapter:
        """Get the appropriate adapter for the output type"""
        if self.detected_type is None:
            self.detected_type = self._detect_output_type(model_output)
            logger.info(f"Auto-detected model output type: {self.detected_type}")

        return self.adapters[self.detected_type]

    def extract_logits(self, model_output: Any) -> torch.Tensor:
        adapter = self._get_adapter(model_output)
        return adapter.extract_logits(model_output)

    def extract_features(self, model_output: Any) -> Optional[torch.Tensor]:
        adapter = self._get_adapter(model_output)
        return adapter.extract_features(model_output)

    def get_prediction_dict(self, model_output: Any) -> Dict[str, torch.Tensor]:
        adapter = self._get_adapter(model_output)
        return adapter.get_prediction_dict(model_output)


# Specialized adapters for specific models
class Meso4Adapter(LogitsAndFeaturesAdapter):
    """Specialized adapter for Meso4 models"""

    pass


class XceptionAdapter(StandardAdapter):
    """Specialized adapter for Xception models"""

    pass


class EfficientNetAdapter(StandardAdapter):
    """Specialized adapter for EfficientNet models"""

    pass


class MesoInceptionAdapter(LogitsAndFeaturesAdapter):
    """Specialized adapter for MesoInception models"""

    pass


def get_adapter_for_model(model_name: str) -> ModelAdapter:
    """
    Factory function to get the appropriate adapter for a model by name

    Args:
        model_name: Name of the model (e.g., 'meso4', 'xception', etc.)

    Returns:
        Appropriate ModelAdapter instance
    """
    model_name = model_name.lower()

    adapter_map = {
        "meso4": Meso4Adapter(),
        "xception": XceptionAdapter(),
        "efficientnet": EfficientNetAdapter(),
        "efficientnetb4": EfficientNetAdapter(),
        "mesoinception": MesoInceptionAdapter(),
        "ucf": UCFAdapter(),
        "capsule": CapsuleNetworkAdapter(),
        "auto": AutoAdapter(),
    }

    if model_name in adapter_map:
        logger.info(f"Using {type(adapter_map[model_name]).__name__} for {model_name}")
        return adapter_map[model_name]
    else:
        logger.warning(f"Unknown model '{model_name}', using AutoAdapter")
        return AutoAdapter()


# Export all adapters for easy import
__all__ = [
    "ModelAdapter",
    "StandardAdapter",
    "LogitsAndFeaturesAdapter",
    "DictOutputAdapter",
    "UCFAdapter",
    "CapsuleNetworkAdapter",
    "CustomFunctionAdapter",
    "AutoAdapter",
    "Meso4Adapter",
    "XceptionAdapter",
    "EfficientNetAdapter",
    "MesoInceptionAdapter",
    "get_adapter_for_model",
]
