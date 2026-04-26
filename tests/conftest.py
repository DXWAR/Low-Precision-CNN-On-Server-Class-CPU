"""
Shared fixtures for the dissertation test suite.

Provides lightweight model/data fixtures that avoid loading the full
ChestX-ray14 dataset or downloading torchxrayvision weights during CI.
Tests use small random tensors and a toy DenseNet to keep execution
under 30 seconds total.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn
import numpy as np

# Ensure the project root is on sys.path so `import newmain` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ─── Determinism ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seed_everything():
    """Fix all random seeds for reproducibility (FR8)."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ─── Tiny DenseNet-like model ──────────────────────────────────────────────────

class _TinyDenseLayer(nn.Module):
    """Minimal DenseLayer that mimics torchxrayvision's concatenation pattern."""

    def __init__(self, in_ch, growth=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=False)

    def forward(self, prev_features):
        if isinstance(prev_features, list):
            x = torch.cat(prev_features, 1)
        else:
            x = prev_features
        x = self.conv(torch.relu(self.bn(x)))
        return x


class _TinyDenseBlock(nn.Module):
    """A 3-layer dense block for testing."""

    def __init__(self, in_ch, growth=4, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(n_layers):
            self.layers.append(_TinyDenseLayer(ch, growth))
            ch += growth

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new = layer(features)
            features.append(new)
        return torch.cat(features, 1)


class TinyDenseNet(nn.Module):
    """
    Toy DenseNet with one dense block + GAP + linear classifier.
    Channel arithmetic: 1 input → 4 after conv0 → 4 + 3*4 = 16 after block → 14 logits.
    """

    def __init__(self, in_ch=1, init_ch=4, growth=4, n_layers=3, n_classes=14):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(in_ch, init_ch, 3, padding=1, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(init_ch))
        self.features.add_module('denseblock1', _TinyDenseBlock(init_ch, growth, n_layers))
        final_ch = init_ch + n_layers * growth
        self.features.add_module('norm5', nn.BatchNorm2d(final_ch))
        self.classifier = nn.Linear(final_ch, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x


@pytest.fixture
def tiny_model():
    """A small DenseNet-like model for fast unit tests."""
    model = TinyDenseNet()
    model.eval()
    return model


@pytest.fixture
def dummy_input():
    """A single 224×224 grayscale image tensor (batch size 1)."""
    return torch.randn(1, 1, 224, 224)


@pytest.fixture
def small_input():
    """A single 32×32 grayscale image tensor for fast tests."""
    return torch.randn(1, 1, 32, 32)


@pytest.fixture
def batch_inputs():
    """A batch of 8 small images."""
    return torch.randn(8, 1, 32, 32)


@pytest.fixture
def pathology_names():
    """The 14 ChestX-ray14 pathology names."""
    return [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
        'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
        'Pneumonia', 'Pneumothorax',
    ]
