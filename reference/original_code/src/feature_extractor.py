"""
feature_extractor.py

This module defines and provides access to feature extractors for hybrid classical-quantum pipelines.

It supports:
- A custom CNN-based extractor designed for low-dimensional latent embeddings (e.g., for quantum circuits).
- Pretrained ResNet18 and DenseNet121 models from torchvision with their classification heads removed.

Main Interfaces:
- CustomCNNFeatureExtractor: A lightweight CNN for extracting fixed-size feature vectors.
- get_feature_extractor(): A unified loader for 'resnet', 'densenet', and 'custom' extractors.

Usage:
    model, feature_dim = get_feature_extractor("resnet")
    model, feature_dim = get_feature_extractor("custom", output_dim=64)

All models are automatically moved to GPU if available.
"""

import torch
import torch.nn as nn
from torchvision import models


class CustomCNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64):
        super(CustomCNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


def get_feature_extractor(extractor_type="resnet", output_dim=None):
    """
    Unified feature extractor loader.
    Args:
        extractor_type (str): 'resnet', 'densenet', or 'custom'.
        output_dim (int, optional): Required for 'custom'.
    Returns:
        model (nn.Module): Feature extractor without classification head.
        feature_dim (int): Output dimensionality.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if extractor_type == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
        feature_dim = 512

    elif extractor_type == "densenet":
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Identity()
        feature_dim = 1024

    elif extractor_type == "custom":
        if output_dim is None:
            raise ValueError("Custom extractor requires output_dim.")
        model = CustomCNNFeatureExtractor(output_dim=output_dim)
        feature_dim = output_dim

    else:
        raise ValueError(f"Unknown extractor_type: {extractor_type}")

    model = model.to(device)
    return model, feature_dim
