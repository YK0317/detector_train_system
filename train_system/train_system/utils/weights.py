"""
Weight utilities for the train system

Provides utilities for working with trained model weights.
"""

import os
from pathlib import Path
from typing import Optional, List, Union
import logging


class WeightManager:
    """Utility class for managing model weights"""

    @staticmethod
    def find_weight_files(
        directory: Union[str, Path], pattern: str = "*checkpoint*"
    ) -> List[Path]:
        """
        Find weight files in a directory

        Args:
            directory: Directory to search
            pattern: File pattern to match

        Returns:
            List of weight file paths
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        weight_files = []
        for ext in ["*.pt", "*.pth"]:
            weight_files.extend(directory.glob(f"{pattern}.{ext.split('.')[-1]}"))

        return sorted(weight_files)

    @staticmethod
    def find_best_checkpoint(directory: Union[str, Path]) -> Optional[Path]:
        """
        Find the best checkpoint file in a directory

        Args:
            directory: Directory to search

        Returns:
            Path to best checkpoint or None if not found
        """
        directory = Path(directory)

        # Look for best checkpoint files
        for ext in ["pt", "pth"]:
            best_file = directory / f"best_checkpoint.{ext}"
            if best_file.exists():
                return best_file

        return None

    @staticmethod
    def find_latest_checkpoint(directory: Union[str, Path]) -> Optional[Path]:
        """
        Find the latest checkpoint file in a directory

        Args:
            directory: Directory to search

        Returns:
            Path to latest checkpoint or None if not found
        """
        directory = Path(directory)

        # Look for latest checkpoint files
        for ext in ["pt", "pth"]:
            latest_file = directory / f"last_checkpoint.{ext}"
            if latest_file.exists():
                return latest_file

        return None

    @staticmethod
    def auto_find_checkpoint(
        directory: Union[str, Path], prefer_best: bool = True
    ) -> Optional[Path]:
        """
        Automatically find the best available checkpoint

        Args:
            directory: Directory to search
            prefer_best: Whether to prefer best over latest checkpoint

        Returns:
            Path to checkpoint or None if not found
        """
        if prefer_best:
            # Try best first, then latest
            checkpoint = WeightManager.find_best_checkpoint(directory)
            if checkpoint:
                return checkpoint
            return WeightManager.find_latest_checkpoint(directory)
        else:
            # Try latest first, then best
            checkpoint = WeightManager.find_latest_checkpoint(directory)
            if checkpoint:
                return checkpoint
            return WeightManager.find_best_checkpoint(directory)

    @staticmethod
    def convert_weight_format(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = "pt",
    ) -> bool:
        """
        Convert weight file from one format to another

        Args:
            input_path: Input weight file path
            output_path: Output weight file path
            target_format: Target format ('pt' or 'pth')

        Returns:
            True if successful, False otherwise
        """
        try:
            import torch

            input_path = Path(input_path)
            output_path = Path(output_path)

            if not input_path.exists():
                logging.error(f"Input file not found: {input_path}")
                return False

            # Load the checkpoint
            checkpoint = torch.load(input_path, map_location="cpu")

            # Ensure output has correct extension
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{target_format}")
            elif output_path.suffix != f".{target_format}":
                output_path = output_path.with_suffix(f".{target_format}")

            # Save in new format
            torch.save(checkpoint, output_path)
            logging.info(f"Converted {input_path} -> {output_path}")
            return True

        except Exception as e:
            logging.error(f"Failed to convert weights: {e}")
            return False

    @staticmethod
    def get_weight_info(weight_path: Union[str, Path]) -> dict:
        """
        Get information about a weight file

        Args:
            weight_path: Path to weight file

        Returns:
            Dictionary with weight file information
        """
        try:
            import torch

            weight_path = Path(weight_path)
            if not weight_path.exists():
                return {"error": f"File not found: {weight_path}"}

            checkpoint = torch.load(weight_path, map_location="cpu")

            info = {
                "file_path": str(weight_path),
                "file_size_mb": weight_path.stat().st_size / (1024 * 1024),
                "format": weight_path.suffix,
                "keys": (
                    list(checkpoint.keys())
                    if isinstance(checkpoint, dict)
                    else ["state_dict"]
                ),
            }

            if isinstance(checkpoint, dict):
                if "epoch" in checkpoint:
                    info["epoch"] = checkpoint["epoch"]
                if "best_val_acc" in checkpoint:
                    info["best_val_acc"] = checkpoint["best_val_acc"]
                if "config" in checkpoint:
                    info["has_config"] = True

            return info

        except Exception as e:
            return {"error": f"Failed to load weights: {e}"}


def find_weights_for_model(
    model_name: str, base_dir: str = "training_output", model_type: str = "auto"
) -> Optional[Path]:
    """
    Find weight files for a specific model

    Args:
        model_name: Name of the model/experiment
        base_dir: Base directory to search
        model_type: Type of model ("auto", "yolo", "pytorch")

    Returns:
        Path to weight file or None
    """
    base_path = Path(base_dir)

    # Look for experiment directory
    model_dir = base_path / model_name
    if model_dir.exists():
        # For YOLO models, prefer YOLO-compatible weights
        if model_type == "yolo" or (
            model_type == "auto" and "yolo" in model_name.lower()
        ):
            # Look for YOLO-compatible weights first
            yolo_weights = model_dir.glob("yolo_*.pt")
            for weight in yolo_weights:
                if weight.exists():
                    return weight

            # Fall back to regular YOLO format (.pt)
            for pattern in ["*best*.pt", "*checkpoint*.pt"]:
                weights = model_dir.glob(pattern)
                for weight in weights:
                    return weight

        # For regular PyTorch models or fallback
        return WeightManager.auto_find_checkpoint(model_dir)

    # Look for pattern in all subdirectories
    for subdir in base_path.iterdir():
        if subdir.is_dir() and model_name.lower() in subdir.name.lower():
            checkpoint = WeightManager.auto_find_checkpoint(subdir)
            if checkpoint:
                return checkpoint

    return None


# CLI utility functions
def list_available_weights(base_dir: str = "training_output"):
    """List all available weight files"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return

    print(f"Available weights in {base_dir}:")
    print("=" * 50)

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            weights = WeightManager.find_weight_files(subdir)
            if weights:
                print(f"\n📁 {subdir.name}:")
                for weight in weights:
                    info = WeightManager.get_weight_info(weight)
                    size_mb = info.get("file_size_mb", 0)
                    epoch = info.get("epoch", "N/A")
                    acc = info.get("best_val_acc", "N/A")
                    print(
                        f"  • {weight.name} ({size_mb:.1f}MB, epoch: {epoch}, acc: {acc})"
                    )


if __name__ == "__main__":
    # Demo usage
    print("Weight Manager Demo")
    list_available_weights()
