#!/usr/bin/env python3
"""
Standalone Meso4 Deepfake Detector
Based on DeepfakeBench implementation
Author: AI Assistant
Date: July 28, 2025

This script provides a standalone implementation of the Meso4 model for deepfake detection.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Meso4(nn.Module):
    """
    Meso4 CNN architecture for deepfake detection
    Based on the original MesoNet paper by Afchar et al.
    """

    def __init__(self, num_classes=2, inc=3):
        super(Meso4, self).__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(inc, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)

        # Pooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        # Fully connected layers
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, self.num_classes)

    def features(self, x):
        """Extract features from input"""
        x = self.conv1(x)  # (8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)  # (8, 128, 128)

        x = self.conv2(x)  # (8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool1(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)  # (16, 8, 8)

        x = x.view(x.size(0), -1)  # Flatten to (Batch, 16*8*8)
        return x

    def classifier(self, features):
        """Classify features"""
        out = self.dropout(features)
        out = self.fc1(out)  # (Batch, 16)
        out = self.leakyrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self, x):
        """Forward pass"""
        features = self.features(x)
        out = self.classifier(features)
        return out, features


class Meso4Detector:
    """Standalone Meso4 Deepfake Detector"""

    def __init__(self, model_path, device=None, confidence_threshold=0.5):
        """
        Initialize the detector

        Args:
            model_path (str): Path to the trained model weights
            device (str): Device to run inference on ('cpu' or 'cuda')
            confidence_threshold (float): Confidence threshold for classification
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.confidence_threshold = confidence_threshold

        # Load model
        self.model = self._load_model(model_path)

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        logger.info(f"Meso4 Detector initialized on {self.device}")

    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize model
        model = Meso4(num_classes=2, inc=3)

        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Remove 'backbone.' prefix if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    new_key = key.replace("backbone.", "")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Model loaded from: {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        if isinstance(image_path, str):
            # Load from path
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            # Convert numpy array to PIL
            image = Image.fromarray(image_path).convert("RGB")
        else:
            # Assume it's already a PIL image
            image = image_path.convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)

    def predict(self, image_input):
        """
        Predict whether an image is real or fake

        Args:
            image_input: Path to image file, numpy array, or PIL Image

        Returns:
            dict: Prediction results containing label, confidence, and probabilities
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_input)

            # Run inference
            with torch.no_grad():
                outputs, features = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence_scores = probabilities.cpu().numpy()[0]

                # Get prediction
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = confidence_scores[predicted_class]

                # Determine label
                is_fake = predicted_class == 1
                label = "FAKE" if is_fake else "REAL"

                return {
                    "label": label,
                    "confidence": float(confidence),
                    "is_fake": is_fake,
                    "probabilities": {
                        "real": float(confidence_scores[0]),
                        "fake": float(confidence_scores[1]),
                    },
                    "raw_output": outputs.cpu().numpy()[0].tolist(),
                }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "is_fake": None,
                "probabilities": {"real": 0.0, "fake": 0.0},
                "error": str(e),
            }

    def predict_batch(self, image_paths):
        """Predict on a batch of images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results

    def evaluate_video(self, video_path, frame_interval=30):
        """
        Evaluate a video by sampling frames

        Args:
            video_path (str): Path to video file
            frame_interval (int): Interval between sampled frames

        Returns:
            dict: Overall video prediction and frame-by-frame results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_results = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.predict(frame_rgb)
                    result["frame_idx"] = frame_idx
                    frame_results.append(result)

                frame_idx += 1

            cap.release()

            # Aggregate results
            if frame_results:
                fake_count = sum(1 for r in frame_results if r["is_fake"])
                total_frames = len(frame_results)
                fake_ratio = fake_count / total_frames

                avg_fake_confidence = np.mean(
                    [r["probabilities"]["fake"] for r in frame_results]
                )

                overall_label = "FAKE" if fake_ratio > 0.5 else "REAL"

                return {
                    "video_path": video_path,
                    "overall_label": overall_label,
                    "fake_ratio": fake_ratio,
                    "avg_fake_confidence": float(avg_fake_confidence),
                    "total_frames_analyzed": total_frames,
                    "frame_results": frame_results,
                }
            else:
                return {"error": "No frames could be processed"}

        except Exception as e:
            logger.error(f"Video evaluation error: {e}")
            return {"error": str(e)}


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Meso4 Deepfake Detector")
    parser.add_argument(
        "--model",
        type=str,
        default="trained/meso4_best.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or video"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument("--video", action="store_true", help="Process as video file")

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = Meso4Detector(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.threshold,
        )

        if args.video:
            # Process video
            result = detector.evaluate_video(args.input)
            print("\n🎬 Video Analysis Results:")
            print(f"📁 File: {args.input}")
            if "error" not in result:
                print(f"🎯 Overall Prediction: {result['overall_label']}")
                print(f"📊 Fake Ratio: {result['fake_ratio']:.2%}")
                print(
                    f"🔍 Average Fake Confidence: {result['avg_fake_confidence']:.4f}"
                )
                print(f"📸 Frames Analyzed: {result['total_frames_analyzed']}")
            else:
                print(f"❌ Error: {result['error']}")
        else:
            # Process single image
            result = detector.predict(args.input)
            print("\n🖼️ Image Analysis Results:")
            print(f"📁 File: {args.input}")
            print(f"🎯 Prediction: {result['label']}")
            print(f"🔍 Confidence: {result['confidence']:.4f}")
            print(
                f"📊 Probabilities: Real={result['probabilities']['real']:.4f}, Fake={result['probabilities']['fake']:.4f}"
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
