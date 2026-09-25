"""CIFAR-10 via torchvision: download automático, augmentation e DataLoaders."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def cifar10_loaders(
    data_dir: str, batch_size: int, pin_memory: bool, num_workers: int = 2
) -> tuple[DataLoader, DataLoader]:
    normalize = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *normalize])
    test_tf = transforms.Compose(normalize)

    # download=True é no-op quando o volume pah-cifar10 já tem os arquivos
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return DataLoader(train_set, shuffle=True, **common), DataLoader(test_set, shuffle=False, **common)
