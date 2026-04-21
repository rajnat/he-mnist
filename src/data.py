"""MNIST loading and preprocessing."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_dataloaders(data_dir: str = "./data", batch_size: int = 64, train_subset: int = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if train_subset:
        train_ds = Subset(train_ds, range(train_subset))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_flat_test_samples(n: int = 10, data_dir: str = "./data"):
    """Return n test images as flat numpy vectors (784,) for HE encryption."""
    import numpy as np
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    images, labels = [], []
    for i in range(n):
        img, label = test_ds[i]
        images.append(img.numpy().flatten().tolist())
        labels.append(label)
    return images, labels
