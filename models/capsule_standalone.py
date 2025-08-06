#!/usr/bin/env python3
"""
Standalone Capsule Network Deepfake Detector
Based on DeepfakeBench implementation
Author: AI Assistant
Date: July 28, 2025

This script provides a standalone implementation of the Capsule Network model for deepfake detection.
Based on the paper: "Capsule-forensics: Using capsule networks to detect forged images and videos"
by Nguyen et al. (ICASSP 2019)
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
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class View(nn.Module):
    """View layer for reshaping tensors"""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class StatsNet(nn.Module):
    """Statistics Network that computes mean and std of feature maps"""

    def __init__(self):
        super(StatsNet, self).__init__()

    def forward(self, x):
        x = x.view(x.data.shape[0], x.data.shape[1], x.data.shape[2] * x.data.shape[3])
        mean = torch.mean(x, 2)
        std = torch.std(x, 2)
        return torch.stack((mean, std), dim=1)


class VggExtractor(nn.Module):
    """VGG19 feature extractor (first 18 layers)"""

    def __init__(self, train=False):
        super(VggExtractor, self).__init__()
        self.vgg_1 = self.Vgg(models.vgg19(pretrained=True), 0, 18)
        if train:
            self.vgg_1.train(mode=True)
            self.freeze_gradient()
        else:
            self.vgg_1.eval()

    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin : (end + 1)])
        return features

    def freeze_gradient(self, begin=0, end=9):
        for i in range(begin, end + 1):
            self.vgg_1[i].requires_grad = False

    def forward(self, input):
        return self.vgg_1(input)


class FeatureExtractor(nn.Module):
    """Feature Extractor with multiple capsules"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.NO_CAPS = 10
        self.capsules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    StatsNet(),
                    nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(8),
                    nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(1),
                    View(-1, 8),
                )
                for _ in range(self.NO_CAPS)
            ]
        )

    def squash(self, tensor, dim):
        """Squashing function for capsules"""
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)
        return self.squash(output, dim=-1)


class RoutingLayer(nn.Module):
    """Dynamic Routing Layer for Capsule Network"""

    def __init__(
        self, num_input_capsules, num_output_capsules, data_in, data_out, num_iterations
    ):
        super(RoutingLayer, self).__init__()
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(
            torch.randn(num_output_capsules, num_input_capsules, data_out, data_in)
        )

    def squash(self, tensor, dim):
        """Squashing function for capsules"""
        squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm))

    def forward(self, x, random=False, dropout=0.0):
        # x[b, data, in_caps]
        x = x.transpose(2, 1)
        # x[b, in_caps, data]

        if random:
            noise = torch.Tensor(0.01 * torch.randn(*self.route_weights.size())).to(
                self.route_weights.device
            )
            route_weights = self.route_weights + noise
        else:
            route_weights = self.route_weights

        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]
        priors = priors.transpose(1, 0)
        # priors[b, out_caps, in_caps, data_out, 1]

        if dropout > 0.0:
            drop = torch.Tensor(
                torch.FloatTensor(*priors.size()).bernoulli(1.0 - dropout)
            ).to(priors.device)
            priors = priors * drop

        logits = torch.Tensor(torch.zeros(*priors.size())).to(priors.device)
        # logits[b, out_caps, in_caps, data_out, 1]

        num_iterations = self.num_iterations

        for i in range(num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)

            if i != self.num_iterations - 1:
                delta_logits = priors * outputs
                logits = logits + delta_logits

        # outputs[b, out_caps, 1, data_out, 1]
        outputs = outputs.squeeze()

        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        # outputs[b, data_out, out_caps]

        return outputs


class CapsuleNetwork(nn.Module):
    """Complete Capsule Network for Deepfake Detection"""

    def __init__(self, num_classes=2):
        super(CapsuleNetwork, self).__init__()
        self.num_classes = num_classes
        self.NO_CAPS = 10

        # Components
        self.vgg_ext = VggExtractor()
        self.fea_ext = FeatureExtractor()
        self.routing_stats = RoutingLayer(
            num_input_capsules=self.NO_CAPS,
            num_output_capsules=self.num_classes,
            data_in=8,
            data_out=4,
            num_iterations=2,
        )

        # Initialize weights
        self.fea_ext.apply(self.weights_init)

    def weights_init(self, m):
        """Initialize weights"""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def features(self, x):
        """Extract features"""
        vgg_features = self.vgg_ext(x)
        capsule_features = self.fea_ext(vgg_features)
        return capsule_features

    def classifier(self, features):
        """Classify features using capsule routing"""
        z = self.routing_stats(features, random=False, dropout=0.0)
        # z[b, data, out_caps]
        classes = F.softmax(z, dim=-1)
        class_ = classes.detach()
        class_ = class_.mean(dim=1)
        return classes, class_

    def forward(self, x):
        """Forward pass"""
        features = self.features(x)
        classes, class_ = self.classifier(features)
        return classes, class_, features


class CapsuleDetector:
    """Standalone Capsule Network Deepfake Detector"""

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
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )

        logger.info(f"Capsule Network Detector initialized on {self.device}")

    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize model
        model = CapsuleNetwork(num_classes=2)

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

            # Load state dict
            model.load_state_dict(state_dict, strict=False)
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
                classes, class_, features = self.model(image_tensor)

                # Get prediction probabilities
                probabilities = torch.softmax(class_, dim=1)
                confidence_scores = probabilities.cpu().numpy()[0]

                # Get prediction
                predicted_class = torch.argmax(class_, dim=1).item()
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
                    "raw_output": class_.cpu().numpy()[0].tolist(),
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
    parser = argparse.ArgumentParser(description="Capsule Network Deepfake Detector")
    parser.add_argument(
        "--model",
        type=str,
        default="trained/capsule_best.pth",
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
        detector = CapsuleDetector(
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
