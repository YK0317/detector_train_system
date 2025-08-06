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


class XceptionBackbone(nn.Module):
    """Xception backbone for UCF"""

    def __init__(self, num_classes=2, inc=3, dropout=False, mode="adjust_channel"):
        super(XceptionBackbone, self).__init__()
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

        # Channel adjustment for UCF
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

        # Only apply adjust_channel if mode is 'adjust_channel'
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)

        return x


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        x = self.conv2d(x)
        return x


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


class UCFStandalone(nn.Module):
    def __init__(self, num_classes=2, inc=3, encoder_feat_dim=2048):
        super(UCFStandalone, self).__init__()
        self.num_classes = num_classes
        self.encoder_feat_dim = (
            encoder_feat_dim  # 2048 (from Xception without adjust_channel)
        )
        self.half_fingerprint_dim = 1024  # This is what the checkpoint shows

        # Two Xception encoders for forgery and content features
        # Use Original mode (not adjust_channel) to get 2048 features
        self.encoder_f = XceptionBackbone(
            num_classes=num_classes, inc=inc, mode="Original"
        )
        self.encoder_c = XceptionBackbone(
            num_classes=num_classes, inc=inc, mode="Original"
        )

        # Heads for specific and shared features
        # Based on checkpoint analysis, both heads output 2 classes
        self.head_spe = Head(
            in_f=self.half_fingerprint_dim,  # 1024
            hidden_dim=self.encoder_feat_dim,  # 2048
            out_f=self.num_classes,  # 2 (from checkpoint analysis)
        )
        self.head_sha = Head(
            in_f=self.half_fingerprint_dim,  # 1024
            hidden_dim=self.encoder_feat_dim,  # 2048
            out_f=self.num_classes,  # 2
        )

        # Blocks for splitting features - these take encoder_feat_dim (2048) as input
        self.block_spe = Conv2d1x1(
            in_f=self.encoder_feat_dim,  # 2048 (from Xception features)
            hidden_dim=1024,  # 1024 (as shown in checkpoint)
            out_f=1024,  # 1024 (as shown in checkpoint)
        )
        self.block_sha = Conv2d1x1(
            in_f=self.encoder_feat_dim,  # 2048 (from Xception features)
            hidden_dim=1024,  # 1024 (as shown in checkpoint)
            out_f=1024,  # 1024 (as shown in checkpoint)
        )

        # We don't need the conditional GAN for inference-only standalone
        # self.con_gan = Conditional_UNet()  # Only needed for training

    def features(self, x):
        # Extract forgery and content features
        f_all = self.encoder_f.features(x)
        c_all = self.encoder_c.features(x)
        return {"forgery": f_all, "content": c_all}

    def classifier(self, features):
        # Split features into specific and shared
        f_spe = self.block_spe(features)  # 2048 -> 1024
        f_share = self.block_sha(features)  # 2048 -> 1024
        return f_spe, f_share

    def forward(self, x):
        # Extract features
        features = self.features(x)
        forgery_features = features["forgery"]  # Shape: [batch, 2048, H, W]

        # Get specific and shared features
        f_spe, f_share = self.classifier(forgery_features)  # Each: [batch, 1024, H, W]

        # For inference, we only use the shared features for final classification
        out_sha, sha_feat = self.head_sha(f_share)

        return out_sha


class UCFDetector:
    def __init__(self, model_path, device="auto"):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transform()
        print(f"UCF Detector initialized on {self.device}")
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
        model = UCFStandalone()
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
    parser = argparse.ArgumentParser(description="UCF Standalone Deepfake Detector")
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\mingw\Desktop\training\trained\ucf_best.pth",
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

    detector = UCFDetector(model_path=args.model_path, device=args.device)

    if args.image_path:
        prediction, fake_prob, real_prob = detector.predict_single(args.image_path)
        print(f"Image: {args.image_path}")
        print(f"Prediction: {prediction}")
        print(f"Fake probability: {fake_prob:.4f}")
        print(f"Real probability: {real_prob:.4f}")
    else:
        print("Please specify --image_path for single image prediction")
