"""
Simple image encoder that doesn't require torchvision.
Uses basic CNN layers instead of ResNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleImageEncoder(nn.Module):
    """Simple CNN encoder for images (no torchvision dependency)."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        feature_dim: int = 256,
    ):
        """
        Args:
            image_size: (height, width) of input images
            feature_dim: Output feature dimension
        """
        super().__init__()
        
        # Simple CNN encoder
        # Input: (B, C, H, W) where C=3, H=224, W=224
        self.encoder = nn.Sequential(
            # First block: 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth block: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling: 7x7 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Projection to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T, C, H, W) or (B*T, C, H, W) image sequence
        Returns:
            features: (B, T, feature_dim) or (B*T, feature_dim)
        """
        original_shape = images.shape
        if images.ndim == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            B, T = images.shape[:2]
            images = images.reshape(B * T, *images.shape[2:])
            reshape_back = True
        else:
            reshape_back = False
        
        # Extract features
        features = self.encoder(images)  # (B*T, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, 512)
        features = self.projection(features)  # (B*T, feature_dim)
        
        if reshape_back:
            features = features.reshape(B, T, -1)  # (B, T, feature_dim)
        
        return features

