"""CIFAR-10 dataset loading utilities.

Provides torchvision transforms and DataLoaders for training and
validation, matching the preprocessing expected by ``src/model.py`` and
``src/serve.py``.
"""

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 per-channel mean/std (computed over the training set).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return the torchvision transform pipeline for CIFAR-10.

    Args:
        train: If True, apply data augmentation (random crop/flip) in
            addition to normalization. If False, only resize/normalize
            (used for validation and inference).

    Returns:
        A ``transforms.Compose`` pipeline producing normalized tensors.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str, batch_size: int = 64, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders for CIFAR-10.

    Downloads the dataset into ``data_dir`` if it is not already present.

    Args:
        data_dir: Directory to store/read the CIFAR-10 dataset.
        batch_size: Batch size for both loaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        A ``(train_loader, val_loader)`` tuple.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
