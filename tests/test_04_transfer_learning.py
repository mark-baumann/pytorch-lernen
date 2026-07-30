"""Tests für 04_transfer_learning.py — ResNet18 Fine-Tuning."""

import torch
from torch import nn
from torchvision import models


class TestResNetTransfer:
    def test_resnet18_loads(self):
        """ResNet18 mit ImageNet-Gewichten sollte ladbar sein."""
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        assert isinstance(model, nn.Module)

    def test_freeze_and_replace_fc(self):
        """FC-Layer ersetzen und einfrieren testen."""
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Alle Layer einfrieren
        for param in model.parameters():
            param.requires_grad = False

        # FC ersetzen
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

        # Nur FC sollte trainierbar sein
        trainable = [p.requires_grad for p in model.parameters()]
        trainable_count = sum(trainable)
        # Nur bias + weights vom neuen FC-Layer
        assert trainable_count == 2

    def test_forward_pass(self):
        """Forward-Pass mit zufälligem Input."""
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_unfreeze_all(self):
        """Alle Layer auftauen."""
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

        for param in model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 1_000_000  # ResNet18 hat ~11M Parameter
