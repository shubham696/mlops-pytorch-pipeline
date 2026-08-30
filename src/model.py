"""Model factory for the classification pipeline.

Exposes ``get_model`` which builds a ``torch.nn.Module`` for a given
architecture name, so training and serving code can share a single
source of truth for model construction.
"""

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHITECTURES = ("resnet18", "simple_cnn")


class SimpleCNN(nn.Module):
    """A small, dependency-free CNN for CIFAR-10-sized (32x32) images.

    Useful as a lightweight fallback when a full ResNet is unnecessary
    (e.g. quick smoke tests, CPU-only environments).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(
    architecture: str, num_classes: int, pretrained: bool = False
) -> nn.Module:
    """Build a model for the given architecture.

    Args:
        architecture: One of ``SUPPORTED_ARCHITECTURES`` (currently
            ``"resnet18"`` or ``"simple_cnn"``).
        num_classes: Number of output classes for the final layer.
        pretrained: If True and ``architecture`` supports it, initialize
            from pretrained (ImageNet) weights. Defaults to False so the
            model can be built offline without downloading weights.

    Returns:
        An initialized ``torch.nn.Module``.

    Raises:
        ValueError: If ``architecture`` is not recognized.
    """
    arch = architecture.lower()
    if arch == "resnet18":
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        f"Supported architectures: {SUPPORTED_ARCHITECTURES}"
    )
