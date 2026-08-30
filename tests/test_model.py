"""Tests for src/model.py and src/dataset.py transforms.

These tests avoid GPU dependencies and the real CIFAR-10 dataset: they
only exercise model construction/forward passes on dummy tensors and
transform pipelines on dummy PIL images.
"""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from dataset import get_transforms  # noqa: E402
from model import get_model  # noqa: E402


def test_get_model_resnet18_output_shape():
    model = get_model("resnet18", 10)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_get_model_simple_cnn_output_shape():
    model = get_model("simple_cnn", 10)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)


def test_get_model_custom_num_classes():
    model = get_model("resnet18", 5)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (1, 5)


def test_get_model_unknown_architecture_raises_value_error():
    with pytest.raises(ValueError):
        get_model("not_a_real_architecture", 10)


def test_get_transforms_eval_returns_compose():
    eval_transforms = get_transforms(train=False)
    assert isinstance(eval_transforms, transforms.Compose)


def test_get_transforms_eval_output_shape():
    eval_transforms = get_transforms(train=False)
    dummy_image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    output = eval_transforms(dummy_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)


def test_get_transforms_train_output_shape():
    train_transforms = get_transforms(train=True)
    dummy_image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    output = train_transforms(dummy_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 32, 32)
