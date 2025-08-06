"""
Standalone BLIP-based Deepfake Detection Model
Self-contained implementation that doesn't require the full BLIP codebase
"""

import math
import os
import warnings
from functools import partial
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Utility functions
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization (from PyTorch)"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal initialization"""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP module for Vision Transformer"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_grad_checkpointing=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.use_grad_checkpointing = use_grad_checkpointing

    def forward(self, x, register_blk=-1):
        if self.use_grad_checkpointing:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer implementation"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_grad_checkpointing=False,
        ckpt_layer=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_grad_checkpointing=(
                        use_grad_checkpointing and i >= depth - ckpt_layer
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[0]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x, register_blk=-1):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)
        return x


def download_file(url: str, filename: str):
    """Download file from URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


class BLIPDeepfakeDetector(nn.Module):
    """
    Standalone BLIP-based Deepfake Detection Model

    This is a self-contained implementation that uses a Vision Transformer backbone
    similar to BLIP for deepfake detection without requiring the full BLIP codebase.
    """

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        vit_arch: str = "base",
        num_classes: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_grad_checkpointing: bool = False,
    ):
        """
        Initialize BLIP Deepfake Detector

        Args:
            img_size: Input image size
            patch_size: Patch size for ViT
            vit_arch: ViT architecture ('base' or 'large')
            num_classes: Number of classes (2 for binary classification)
            dropout: Dropout rate for classifier head
            drop_path_rate: Drop path rate for regularization
            use_grad_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # Define ViT architecture parameters
        if vit_arch == "base":
            embed_dim = 768
            depth = 12
            num_heads = 12
        elif vit_arch == "large":
            embed_dim = 1024
            depth = 24
            num_heads = 16
        else:
            raise ValueError(f"Unsupported ViT architecture: {vit_arch}")

        # Vision Transformer backbone
        self.visual_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            use_grad_checkpointing=use_grad_checkpointing,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        """
        Forward pass

        Args:
            x: Input images (B, 3, H, W)
            targets: Ground truth labels (B,) - 0 for real, 1 for fake
            return_features: Whether to return intermediate features

        Returns:
            If training and targets provided: loss tensor
            If inference: logits tensor (B, num_classes)
            If return_features: (logits, features) tuple
        """
        # Extract visual features
        visual_features = self.visual_encoder(x)  # (B, num_patches+1, embed_dim)

        # Use CLS token for classification
        cls_features = visual_features[:, 0, :]  # (B, embed_dim)

        # Classification
        logits = self.classifier(cls_features)  # (B, num_classes)

        if return_features:
            return logits, cls_features

        # Training mode with targets
        if self.training and targets is not None:
            loss = F.cross_entropy(logits, targets)
            return loss

        return logits

    def predict(self, x: torch.Tensor, return_probs: bool = True):
        """
        Make predictions on input images

        Args:
            x: Input images (B, 3, H, W)
            return_probs: Whether to return probabilities (True) or logits (False)

        Returns:
            Predictions (probabilities or logits)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if return_probs:
                return F.softmax(logits, dim=1)
            return logits

    def predict_single(self, x: torch.Tensor):
        """
        Predict single image and return detailed results

        Args:
            x: Single image tensor (1, 3, H, W) or (3, H, W)

        Returns:
            Dictionary with prediction results
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        probs = self.predict(x, return_probs=True)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs, dim=1)[0].item()

        return {
            "prediction": "fake" if pred_class == 1 else "real",
            "confidence": confidence,
            "fake_probability": probs[0, 1].item(),
            "real_probability": probs[0, 0].item(),
            "predicted_class": pred_class,
        }

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained weights

        Args:
            checkpoint_path: Path to checkpoint file or URL
            strict: Whether to strictly enforce key matching
        """
        if checkpoint_path.startswith("http"):
            # Download if URL
            filename = os.path.basename(urlparse(checkpoint_path).path)
            if not os.path.exists(filename):
                print(f"Downloading pretrained weights from {checkpoint_path}")
                download_file(checkpoint_path, filename)
            checkpoint_path = filename

        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Filter only visual encoder weights if loading from BLIP checkpoint
        visual_encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith("visual_encoder"):
                # Remove 'visual_encoder.' prefix
                new_key = key.replace("visual_encoder.", "")
                visual_encoder_dict[f"visual_encoder.{new_key}"] = value
            elif (
                "patch_embed" in key
                or "pos_embed" in key
                or "cls_token" in key
                or "blocks" in key
                or "norm" in key
            ):
                # Direct ViT weights
                visual_encoder_dict[f"visual_encoder.{key}"] = value

        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(
            visual_encoder_dict, strict=False
        )

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("Pretrained weights loaded successfully!")

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze the visual encoder backbone

        Args:
            freeze: Whether to freeze the backbone
        """
        for param in self.visual_encoder.parameters():
            param.requires_grad = not freeze

        print(f"Visual encoder backbone {'frozen' if freeze else 'unfrozen'}")

    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "img_size": self.img_size,
            "num_classes": self.num_classes,
        }


def create_blip_deepfake_detector(
    vit_arch: str = "base", img_size: int = 384, pretrained: bool = True, **kwargs
) -> BLIPDeepfakeDetector:
    """
    Create a BLIP-based deepfake detector

    Args:
        vit_arch: ViT architecture ('base' or 'large')
        img_size: Input image size
        pretrained: Whether to load pretrained BLIP weights
        **kwargs: Additional arguments for the model

    Returns:
        BLIPDeepfakeDetector model
    """
    model = BLIPDeepfakeDetector(img_size=img_size, vit_arch=vit_arch, **kwargs)

    if pretrained:
        # BLIP pretrained model URLs
        pretrained_urls = {
            "base": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth",
            "large": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth",
        }

        if vit_arch in pretrained_urls:
            try:
                model.load_pretrained_weights(pretrained_urls[vit_arch])
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Proceeding with randomly initialized weights")

    return model


# Example usage and utilities
if __name__ == "__main__":
    # Create model
    model = create_blip_deepfake_detector(
        vit_arch="base", img_size=384, pretrained=False
    )

    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 384, 384)
    dummy_targets = torch.tensor([0, 1])  # real, fake

    # Training mode
    model.train()
    loss = model(dummy_input, dummy_targets)
    print(f"\nTraining loss: {loss.item():.4f}")

    # Inference mode
    model.eval()
    logits = model(dummy_input)
    probs = F.softmax(logits, dim=1)
    print(f"Inference probabilities:\n{probs}")

    # Single image prediction
    single_result = model.predict_single(dummy_input[0])
    print(f"\nSingle image prediction: {single_result}")
