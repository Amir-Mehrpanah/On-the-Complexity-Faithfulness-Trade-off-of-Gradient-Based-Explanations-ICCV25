import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import torch

import paths

registered_datasets = {}


def register_dataset(name):
    def decorator(func):
        # we make sure that the path is set in paths.py
        func.__default_path__ = getattr(paths, f"{name.upper()}_ROOT")
        # we register the function
        registered_datasets[name] = func
        return func

    return decorator


def get_training_and_test_data(dataset, batch_size, **dataset_kwargs):

    if dataset not in registered_datasets:
        raise ValueError(
            f"Dataset {dataset} not found. Available datasets: {registered_datasets.keys()}"
        )
    # dict lookup instead of if-else
    training_data, test_data = registered_datasets[dataset](
        **dataset_kwargs,
    )
    # DATA LOADER
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=2,
    )
    return train_dataloader, test_dataloader


CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


# see b-cos v2 for this
# We are adding to do sensitivity analysis
class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it.
    Results in a [B, 2C, H, W] output. Single image [C, H, W] is also accepted.

    Args:
        dim (int): where to add channels to. Default: -3
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)


@register_dataset("cifar10")
def get_cifar10_dataset(img_size=32, add_inverse=False):

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(img_size, padding=4),
            torchvision.transforms.ToTensor(),
            (
                AddInverse()
                if add_inverse
                else torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ),
        ]
    )

    label_dtype = torch.float32
    label_transform = None

    training_data = datasets.CIFAR10(
        root=paths.CIFAR10_ROOT,
        train=True,
        download=False,
        transform=transform,
        target_transform=label_transform,
    )

    test_data = datasets.CIFAR10(
        root=paths.CIFAR10_ROOT,
        train=False,
        download=False,
        transform=transform,
        target_transform=label_transform,
    )
    return training_data, test_data


@register_dataset("mist")
def get_mnist_dataset(img_size):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
        ]
    )

    label_dtype = torch.float32
    # label_transform = lambda y: torch.tensor([y <= 4, y > 4], dtype=label_dtype)
    label_transform = None

    training_data = datasets.MNIST(
        root=paths.MNIST_ROOT,
        train=True,
        download=False,
        transform=transform,
        target_transform=label_transform,
    )

    test_data = datasets.MNIST(
        root=paths.MNIST_ROOT,
        train=False,
        download=False,
        transform=transform,
        target_transform=label_transform,
    )

    return training_data, test_data
