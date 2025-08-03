import sys
import subprocess
import os
from ultralytics import YOLO
import torch

# ============================================================================
# Check and install required dependencies


def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["ultralytics", "torch"]

for pkg in required_packages:
    install_and_import(pkg)

# ============================================================================
# MODEL 2: YOLO TRAINING-COMPATIBLE VERSION
# ============================================================================

class YOLOStandalone(torch.nn.Module):
    """
    PyTorch-compatible YOLO model for training with the unified wrapper.
    This wraps the YOLO model to make it compatible with standard PyTorch training.
    """
    
    def __init__(self, num_classes=2, model_size='yolov8n', pretrained=True):
        super(YOLOStandalone, self).__init__()
        self.num_classes = num_classes
        self.model_size = model_size
        
        # Simple CNN classifier for training compatibility
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass compatible with PyTorch training"""
        return self.classifier(x)


# ============================================================================
# MODEL 3: YOLO DEPLOYMENT
# ============================================================================

class YOLODetector:
    """YOLO based deepfake detector"""

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained YOLO model"""
        try:
            print("[DEBUG] Loading YOLO model...")
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print("✅ YOLO model loaded successfully")
            else:
                print("❌ YOLO model file not found")
                raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO: {str(e)}")
            raise

    def check(self, image_path, confidence_threshold=0.5):
        """Predict using YOLO"""
        try:
            results = self.model.predict(image_path, imgsz=256, verbose=False)

            if results and hasattr(results[0], 'probs'):
                probs = results[0].probs
                top1_idx = probs.top1
                confidence = probs.top1conf.item()

                # Inverted class mapping
                class_names = {0: 'fake', 1: 'real'}
                prediction = class_names[top1_idx]
                inference_time = results[0].speed['inference']

                # Extract actual probabilities from YOLO output
                probs_data = probs.data.cpu().numpy()
                fake_prob = float(probs_data[0])  # Index 0 = fake
                real_prob = float(probs_data[1])  # Index 1 = real

                return {
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "fake_prob": fake_prob,
                    "real_prob": real_prob,
                    "inference_time": float(inference_time),
                    "model": "YOLO"
                }
            else:
                return {"error": "YOLO prediction failed", "model": "YOLO"}

        except Exception as e:
            return {"error": str(e), "model": "YOLO"}
    
