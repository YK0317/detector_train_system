"""
Adapter for Capsule-Forensics-v2 model

This adapter handles the specific output format of Capsule-Forensics models.
"""

import torch
import torch.nn.functional as F

# Try different import paths for flexibility
try:
    # First, try the direct import that should work in the installed package
    from train_system.adapters import ModelAdapter
except ImportError:
    try:
        # If that doesn't work, try relative import for local development
        from unified_training.adapters import ModelAdapter
    except ImportError:
        # As a fallback, define a compatible adapter interface
        class ModelAdapter:
            """Base adapter class with required methods."""

            def __init__(self, model=None):
                self.model = model

            def extract_logits(self, outputs):
                pass

            def extract_features(self, outputs):
                pass

            def get_predictions(self, outputs):
                pass

            def get_prediction_dict(self, outputs):
                pass


class CapsuleForensicsAdapter(ModelAdapter):
    """
    Adapter for Capsule-Forensics-v2 model outputs

    The Capsule-Forensics-v2 model typically returns a tuple:
    (classes, reconstructions, activations)

    This adapter properly extracts the logits and features from this output format.
    """

    def __init__(self, model=None):
        """
        Initialize the adapter

        Args:
            model: The Capsule-Forensics model (optional)
        """
        self.model = model

    def extract_logits(self, model_output):
        """
        Extract classification logits from model output

        For Capsule-Forensics-v2, the logits (class probabilities) are in the first
        element of the output tuple.

        Args:
            model_output: Output tuple from the model's forward pass

        Returns:
            torch.Tensor: Logits for classification
        """
        # Handle different output formats
        if isinstance(model_output, (list, tuple)):
            if len(model_output) >= 3:
                # Expected format: (classes, reconstructions, activations)
                classes, _, _ = model_output
                return classes
            elif len(model_output) == 2:
                # Format: (classes, reconstructions) or (classes, activations)
                classes, _ = model_output
                return classes
            else:
                # Single output in a tuple/list
                return model_output[0]
        else:
            # Single tensor output
            return model_output

    def extract_features(self, model_output):
        """
        Extract features from model output

        For Capsule-Forensics-v2, we use the reconstructions as features.

        Args:
            model_output: Output tuple from the model's forward pass

        Returns:
            torch.Tensor: Features for interpretation or visualization
        """
        # Handle different output formats
        if isinstance(model_output, (list, tuple)):
            if len(model_output) >= 3:
                # Expected format: (classes, reconstructions, activations)
                _, reconstructions, _ = model_output
                return reconstructions.view(reconstructions.size(0), -1)
            elif len(model_output) == 2:
                # Format: (classes, reconstructions) - use second element
                _, features = model_output
                return features.view(features.size(0), -1)
            else:
                # Single output in a tuple/list
                output = model_output[0]
                return output.view(output.size(0), -1)
        else:
            # Single tensor output - flatten it
            return model_output.view(model_output.size(0), -1)

    def get_predictions(self, model_output):
        """
        Get predictions dictionary from model output

        Creates a dictionary with probabilities, predictions, and logits.

        Args:
            model_output: Output tuple from the model's forward pass

        Returns:
            Dict: Dictionary with prediction data
        """
        # Extract components based on output format
        if isinstance(model_output, (list, tuple)):
            if len(model_output) >= 3:
                # Expected format: (classes, reconstructions, activations)
                classes, reconstructions, activations = model_output
            elif len(model_output) == 2:
                # Format: (classes, reconstructions) or (classes, activations)
                classes, second_output = model_output
                reconstructions = (
                    second_output  # Could be reconstructions or activations
                )
                activations = None
            else:
                # Single output in a tuple/list
                classes = model_output[0]
                reconstructions = None
                activations = None
        else:
            # Single tensor output
            classes = model_output
            reconstructions = None
            activations = None

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(classes, dim=1)

        # Get predicted class indices
        _, pred_idx = torch.max(probs, 1)

        # Create predictions dictionary
        result = {
            "probabilities": probs,
            "predictions": pred_idx,
            "logits": classes,
        }

        # Add optional components if available
        if reconstructions is not None:
            result["reconstructions"] = reconstructions
        if activations is not None:
            result["activations"] = activations

        return result

    def get_prediction_dict(self, model_output):
        """
        Alias for get_predictions to match the abstract method requirement.

        Args:
            model_output: Output tuple from the model's forward pass

        Returns:
            Dict: Dictionary with prediction data
        """
        return self.get_predictions(model_output)
