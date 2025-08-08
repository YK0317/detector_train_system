#!/usr/bin/env python3
"""
Enhanced Standalone Xception Deepfake Detector
Based on train-system implementation with comprehensive deployment features
Author: AI Assistant
Date: August 7, 2025

This script provides a comprehensive standalone implementation of the Xception model for deepfake detection.
Includes batch processing, video analysis, confidence thresholding, and detailed logging.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Union, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution implementation"""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    """Xception Block with residual connections"""
    
    def __init__(
        self,
        in_filters,
        out_filters,
        reps,
        strides=1,
        start_with_relu=True,
        grow_first=True,
    ):
        super(Block, self).__init__()
        
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(
                in_filters, out_filters, 1, stride=strides, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False)
            )
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    in_filters, out_filters, 3, stride=1, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
            
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class XceptionStandalone(nn.Module):
    """
    Xception architecture for deepfake detection
    Based on the original Xception paper by Chollet et al.
    Adapted for binary classification (real vs fake)
    """

    def __init__(self, num_classes=2, inc=3, dropout=0.5, mode="Original"):
        super(XceptionStandalone, self).__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.mode = mode

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # Classifier
        final_channel = 2048
        if self.mode == "adjust_channel_iid":
            final_channel = 512
            self.mode = "adjust_channel"

        if self.dropout_prob > 0:
            self.dropout = nn.Dropout(p=self.dropout_prob)
        else:
            self.dropout = None

        self.last_linear = nn.Linear(final_channel, self.num_classes)

        # Channel adjustment for some modes
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def features(self, x):
        """Extract features from input"""
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        # Exit flow
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)

        return x

    def classifier(self, features):
        """Classify features"""
        # for iid
        if self.mode == "adjust_channel":
            x = features
        else:
            x = self.relu(features)

        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

        if self.dropout is not None:
            x = self.dropout(x)

        self.last_emb = x
        out = self.last_linear(x)
        return out

    def forward(self, x):
        """Forward pass"""
        features = self.features(x)
        out = self.classifier(features)
        return out, features


class XceptionDetector:
    """Enhanced Standalone Xception Deepfake Detector with comprehensive features"""

    def __init__(
        self, 
        model_path: str, 
        device: Optional[str] = None, 
        confidence_threshold: float = 0.5,
        input_size: int = 256,
        normalization_type: str = "standard"
    ):
        """
        Initialize the detector

        Args:
            model_path (str): Path to the trained model weights
            device (str): Device to run inference on ('cpu', 'cuda', or 'auto')
            confidence_threshold (float): Confidence threshold for classification
            input_size (int): Input image size (default: 256)
            normalization_type (str): Type of normalization ('standard' or 'imagenet')
        """
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup image preprocessing
        self.transform = self._setup_transform(normalization_type)
        
        logger.info(f"Xception Detector initialized on {self.device}")
        logger.info(f"Input size: {input_size}x{input_size}")
        logger.info(f"Confidence threshold: {confidence_threshold}")

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device"""
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device_str = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA available: {gpu_name}")
            else:
                device_str = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            device_str = device
            
        return torch.device(device_str)

    def _load_model(self, model_path: str) -> XceptionStandalone:
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize model
        model = XceptionStandalone(num_classes=2, inc=3, dropout=0.5)

        # Load weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    if "best_val_acc" in checkpoint:
                        logger.info(f"Model best validation accuracy: {checkpoint['best_val_acc']:.4f}")
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Handle different key prefixes from training
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Remove common prefixes
                if key.startswith("module."):
                    new_key = key.replace("module.", "")
                elif key.startswith("backbone."):
                    new_key = key.replace("backbone.", "")
                elif key.startswith("model."):
                    new_key = key.replace("model.", "")
                    
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"Model loaded successfully from: {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        model.to(self.device)
        model.eval()
        return model

    def _setup_transform(self, normalization_type: str) -> transforms.Compose:
        """Setup image preprocessing pipeline"""
        transform_list = [
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
        ]
        
        if normalization_type == "imagenet":
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            )
        else:
            # Standard normalization (default)
            transform_list.append(
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5]
                )
            )
        
        return transforms.Compose(transform_list)

    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for inference"""
        if isinstance(image_input, str):
            # Load from path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL
            if image_input.shape[2] == 3:  # BGR to RGB conversion
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_input).convert("RGB")
        else:
            # Assume it's already a PIL image
            image = image_input.convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)

    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
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
                
                # Check confidence threshold
                is_confident = confidence >= self.confidence_threshold

                return {
                    "label": label,
                    "confidence": float(confidence),
                    "is_fake": is_fake,
                    "is_confident": is_confident,
                    "probabilities": {
                        "real": float(confidence_scores[0]),
                        "fake": float(confidence_scores[1]),
                    },
                    "raw_output": outputs.cpu().numpy()[0].tolist(),
                    "features_shape": list(features.shape),
                }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "is_fake": None,
                "is_confident": False,
                "probabilities": {"real": 0.0, "fake": 0.0},
                "error": str(e),
            }

    def predict_batch(self, image_inputs: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict]:
        """Predict on a batch of images"""
        results = []
        logger.info(f"Processing batch of {len(image_inputs)} images...")
        
        for i, image_input in enumerate(image_inputs):
            if i % 10 == 0:
                logger.info(f"Processing image {i+1}/{len(image_inputs)}")
            
            result = self.predict(image_input)
            if isinstance(image_input, str):
                result["image_path"] = image_input
            results.append(result)
            
        return results

    def evaluate_video(
        self, 
        video_path: str, 
        frame_interval: int = 30,
        max_frames: Optional[int] = None,
        output_individual_frames: bool = False
    ) -> Dict:
        """
        Evaluate a video by sampling frames

        Args:
            video_path (str): Path to video file
            frame_interval (int): Interval between sampled frames
            max_frames (int): Maximum number of frames to process
            output_individual_frames (bool): Whether to include individual frame results

        Returns:
            dict: Overall video prediction and frame-by-frame results
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")

            frame_results = []
            frame_idx = 0
            processed_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    if max_frames and processed_frames >= max_frames:
                        break
                        
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.predict(frame_rgb)
                    result["frame_idx"] = frame_idx
                    result["timestamp"] = frame_idx / fps if fps > 0 else 0
                    frame_results.append(result)
                    processed_frames += 1
                    
                    if processed_frames % 10 == 0:
                        logger.info(f"Processed {processed_frames} frames...")

                frame_idx += 1

            cap.release()

            # Aggregate results
            if frame_results:
                # Count predictions
                fake_count = sum(1 for r in frame_results if r["is_fake"])
                total_frames = len(frame_results)
                fake_ratio = fake_count / total_frames

                # Calculate average confidences
                avg_fake_confidence = np.mean([r["probabilities"]["fake"] for r in frame_results])
                avg_real_confidence = np.mean([r["probabilities"]["real"] for r in frame_results])
                
                # Determine overall prediction
                overall_label = "FAKE" if fake_ratio > 0.5 else "REAL"
                overall_confidence = avg_fake_confidence if fake_ratio > 0.5 else avg_real_confidence

                # Calculate confidence metrics
                confident_predictions = sum(1 for r in frame_results if r["is_confident"])
                confidence_ratio = confident_predictions / total_frames

                result = {
                    "video_path": video_path,
                    "overall_label": overall_label,
                    "overall_confidence": float(overall_confidence),
                    "fake_ratio": fake_ratio,
                    "confidence_ratio": confidence_ratio,
                    "avg_fake_confidence": float(avg_fake_confidence),
                    "avg_real_confidence": float(avg_real_confidence),
                    "total_frames_analyzed": total_frames,
                    "total_frames_in_video": frame_count,
                    "video_duration": duration,
                    "fps": fps,
                    "frame_interval": frame_interval,
                }
                
                if output_individual_frames:
                    result["frame_results"] = frame_results
                else:
                    result["frame_results"] = f"{total_frames} frames analyzed (use --output-frames to see details)"

                return result
            else:
                return {"error": "No frames could be processed"}

        except Exception as e:
            logger.error(f"Video evaluation error: {e}")
            return {"error": str(e)}

    def save_results(self, results: Union[Dict, List[Dict]], output_path: str):
        """Save prediction results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(
        description="Enhanced Xception Deepfake Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python xception_deployment.py --model best_model.pth --input image.jpg
  
  # Batch processing
  python xception_deployment.py --model best_model.pth --batch-dir /path/to/images/
  
  # Video analysis
  python xception_deployment.py --model best_model.pth --input video.mp4 --video
  
  # High-confidence predictions only
  python xception_deployment.py --model best_model.pth --input image.jpg --threshold 0.8
        """
    )
    
    # Model and input arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        help="Path to input image or video"
    )
    parser.add_argument(
        "--batch-dir", 
        type=str, 
        help="Directory containing images for batch processing"
    )
    
    # Processing options
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        choices=["auto", "cpu", "cuda"],
        help="Device to use (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for classification"
    )
    parser.add_argument(
        "--input-size", 
        type=int, 
        default=256, 
        help="Input image size"
    )
    parser.add_argument(
        "--normalization", 
        type=str, 
        default="standard", 
        choices=["standard", "imagenet"],
        help="Normalization type"
    )
    
    # Video processing options
    parser.add_argument(
        "--video", 
        action="store_true", 
        help="Process input as video file"
    )
    parser.add_argument(
        "--frame-interval", 
        type=int, 
        default=30, 
        help="Frame sampling interval for video analysis"
    )
    parser.add_argument(
        "--max-frames", 
        type=int, 
        help="Maximum number of frames to process from video"
    )
    parser.add_argument(
        "--output-frames", 
        action="store_true", 
        help="Output individual frame results for videos"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.input and not args.batch_dir:
        parser.error("Either --input or --batch-dir must be specified")

    try:
        # Initialize detector
        detector = XceptionDetector(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.threshold,
            input_size=args.input_size,
            normalization_type=args.normalization
        )

        results = None

        if args.batch_dir:
            # Batch processing
            if not os.path.isdir(args.batch_dir):
                raise ValueError(f"Directory not found: {args.batch_dir}")
                
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(args.batch_dir).glob(f"*{ext}"))
                image_files.extend(Path(args.batch_dir).glob(f"*{ext.upper()}"))
            
            if not image_files:
                raise ValueError(f"No image files found in: {args.batch_dir}")
                
            logger.info(f"Found {len(image_files)} images in {args.batch_dir}")
            
            # Process batch
            start_time = time.time()
            results = detector.predict_batch([str(f) for f in image_files])
            processing_time = time.time() - start_time
            
            # Print summary
            print(f"\n📁 Batch Processing Results:")
            print(f"📂 Directory: {args.batch_dir}")
            print(f"📸 Images processed: {len(results)}")
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"🚀 Average time per image: {processing_time/len(results):.3f}s")
            
            # Count predictions
            fake_count = sum(1 for r in results if r.get("is_fake"))
            real_count = len(results) - fake_count
            confident_count = sum(1 for r in results if r.get("is_confident"))
            
            print(f"🎯 Predictions: {fake_count} FAKE, {real_count} REAL")
            print(f"🎪 Confident predictions: {confident_count}/{len(results)} ({confident_count/len(results)*100:.1f}%)")
            
        elif args.video:
            # Video processing
            if not args.input:
                parser.error("--input must be specified for video processing")
                
            start_time = time.time()
            results = detector.evaluate_video(
                video_path=args.input,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
                output_individual_frames=args.output_frames
            )
            processing_time = time.time() - start_time
            
            # Print results
            print(f"\n🎬 Video Analysis Results:")
            print(f"📁 File: {args.input}")
            
            if "error" not in results:
                print(f"🎯 Overall Prediction: {results['overall_label']}")
                print(f"🔍 Overall Confidence: {results['overall_confidence']:.4f}")
                print(f"📊 Fake Ratio: {results['fake_ratio']:.2%}")
                print(f"🎪 Confidence Ratio: {results['confidence_ratio']:.2%}")
                print(f"📸 Frames Analyzed: {results['total_frames_analyzed']}")
                print(f"🎞️ Total Frames: {results['total_frames_in_video']}")
                print(f"⏱️ Duration: {results['video_duration']:.2f}s")
                print(f"🚀 Processing Time: {processing_time:.2f}s")
            else:
                print(f"❌ Error: {results['error']}")
                
        else:
            # Single image processing
            if not args.input:
                parser.error("--input must be specified for single image processing")
                
            start_time = time.time()
            results = detector.predict(args.input)
            processing_time = time.time() - start_time
            
            # Print results
            print(f"\n🖼️ Image Analysis Results:")
            print(f"📁 File: {args.input}")
            
            if "error" not in results:
                print(f"🎯 Prediction: {results['label']}")
                print(f"🔍 Confidence: {results['confidence']:.4f}")
                print(f"🎪 Confident: {'Yes' if results['is_confident'] else 'No'}")
                print(f"📊 Probabilities:")
                print(f"  • Real: {results['probabilities']['real']:.4f}")
                print(f"  • Fake: {results['probabilities']['fake']:.4f}")
                print(f"🚀 Processing Time: {processing_time:.3f}s")
            else:
                print(f"❌ Error: {results['error']}")

        # Save results if requested
        if args.output and results:
            detector.save_results(results, args.output)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
