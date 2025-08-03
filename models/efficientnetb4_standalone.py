#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Union, List, Tuple
import time
from efficientnet_pytorch import EfficientNet

class EfficientNetB4Standalone(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=False, mode='Original'):
        super(EfficientNetB4Standalone, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode = mode
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4')
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.last_layer = nn.Linear(1792, self.num_classes)
        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
    
    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x
    
    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        self.last_emb = x
        y = self.last_layer(x)
        return y
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EfficientNetB4Detector:
    def __init__(self, model_path, device='auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transform()
        print(f'EfficientNetB4 Detector initialized on {self.device}')
        print(f'Model loaded from: {model_path}')
    
    def _setup_device(self, device):
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f'CUDA available: {torch.cuda.get_device_name(0)}')
            else:
                device = 'cpu'
                print('CUDA not available, using CPU')
        return torch.device(device)
    
    def _load_model(self, model_path):
        model = EfficientNetB4Standalone()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        print(f'Loading model weights from: {model_path}')
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        # Handle different key prefixes from training
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        if any(key.startswith('backbone.') for key in state_dict.keys()):
            state_dict = {key.replace('backbone.', ''): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
    
    def _setup_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def predict_single(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            real_prob = probabilities[0, 0].item()
            fake_prob = probabilities[0, 1].item()
            prediction = 'FAKE' if fake_prob > real_prob else 'REAL'
        return prediction, fake_prob, real_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EfficientNetB4 Standalone Deepfake Detector')
    parser.add_argument('--model_path', type=str, default=r'C:\Users\mingw\Desktop\training\trained\ef4_best.pth', help='Path to the trained model weights')
    parser.add_argument('--image_path', type=str, help='Path to a single image for prediction')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use for inference')
    args = parser.parse_args()
    
    detector = EfficientNetB4Detector(model_path=args.model_path, device=args.device)
    
    if args.image_path:
        prediction, fake_prob, real_prob = detector.predict_single(args.image_path)
        print(f'Image: {args.image_path}')
        print(f'Prediction: {prediction}')
        print(f'Fake probability: {fake_prob:.4f}')
        print(f'Real probability: {real_prob:.4f}')
    else:
        print('Please specify --image_path for single image prediction')
