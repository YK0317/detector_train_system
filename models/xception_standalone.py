#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class SeparableConv2d(nn.Module):
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
    def __init__(self, num_classes=2, inc=3, dropout=False, mode="Original"):
        super(XceptionStandalone, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
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

        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.last_linear = nn.Linear(final_channel, self.num_classes)

        # Channel adjustment for some modes
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def features(self, x):
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
        # for iid
        if self.mode == "adjust_channel":
            x = features
        else:
            x = self.relu(features)

        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

        if self.dropout:
            x = self.dropout_layer(x)

        self.last_emb = x
        out = self.last_linear(x)
        return out

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


class XceptionDetector:
    def __init__(self, model_path, device="auto"):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transform()
        print(f"Xception Detector initialized on {self.device}")
        print(f"Model loaded from: {model_path}")

    def _setup_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        return torch.device(device)

    def _load_model(self, model_path):
        model = XceptionStandalone()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        # Handle different key prefixes from training
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {
                key.replace("module.", ""): value for key, value in state_dict.items()
            }
        if any(key.startswith("backbone.") for key in state_dict.keys()):
            state_dict = {
                key.replace("backbone.", ""): value for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _setup_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def predict_single(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            real_prob = probabilities[0, 0].item()
            fake_prob = probabilities[0, 1].item()
            prediction = "FAKE" if fake_prob > real_prob else "REAL"
        return prediction, fake_prob, real_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xception Standalone Deepfake Detector"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\mingw\Desktop\training\trained\xception_best.pth",
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to a single image for prediction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference",
    )
    args = parser.parse_args()

    detector = XceptionDetector(model_path=args.model_path, device=args.device)

    if args.image_path:
        prediction, fake_prob, real_prob = detector.predict_single(args.image_path)
        print(f"Image: {args.image_path}")
        print(f"Prediction: {prediction}")
        print(f"Fake probability: {fake_prob:.4f}")
        print(f"Real probability: {real_prob:.4f}")
    else:
        print("Please specify --image_path for single image prediction")
