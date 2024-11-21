import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
import torch

from src.utils import DatasetSwitch
from src import paths

registered_datasets = {}


def register_dataset(name):
    def decorator(func):
        str_name = str(name)
        # we make sure that the path is set in paths.py
        func.__root_path__ = getattr(paths, f"{str_name.upper()}_ROOT")
        # we register the function
        registered_datasets[name] = func
        return func

    return decorator


def get_training_and_test_data(
    dataset,
    root_path,
    batch_size,
    num_workers=2,
    prefetch_factor=4,
    **dataset_kwargs,
):

    if dataset not in registered_datasets:
        raise ValueError(
            f"Dataset {dataset} not found. Available datasets: {registered_datasets.keys()}"
        )
    # dict lookup instead of if-else
    training_data, test_data = registered_datasets[dataset](
        root_path=root_path,
        **dataset_kwargs,
    )
    num_classes = len(training_data.classes)
    input_shape = training_data[0][0].shape
    assert input_shape == test_data[0][0].shape

    # DATA LOADER
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader, input_shape, num_classes


CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)
IMAGENETTE_MEAN = (0.485, 0.456, 0.406)
IMAGENETTE_STD = (0.229, 0.224, 0.225)


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


@register_dataset(DatasetSwitch.IMAGENETTE)
def get_imagenette_dataset(root_path, img_size, add_inverse=False, **kwargs):
    img_size = 256 if img_size is None else img_size
    label_transform = None
    training_data = get_imagenette_train(
        root_path,
        img_size,
        add_inverse,
        label_transform,
    )
    test_data = get_imagenette_test(
        root_path,
        img_size,
        add_inverse,
        label_transform,
    )

    return training_data, test_data


def get_imagenette_train(root_path, img_size, add_inverse, label_transform=None):
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            (
                # ablation of Bcos
                AddInverse()
                if add_inverse
                else torchvision.transforms.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STD)
            ),
        ]
    )
    training_data = datasets.Imagenette(
        root=root_path,
        split="train",
        transform=train_transform,
        target_transform=label_transform,
        download=False,
    )

    return training_data


def get_imagenette_test(root_path, img_size, add_inverse, label_transform=None):
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            (
                # ablation of Bcos
                AddInverse()
                if add_inverse
                else torchvision.transforms.Normalize(IMAGENETTE_MEAN, IMAGENETTE_STD)
            ),
        ]
    )
    test_data = datasets.Imagenette(
        root=root_path,
        split="val",
        transform=test_transform,
        target_transform=label_transform,
        download=False,
    )

    return test_data


@register_dataset(DatasetSwitch.CIFAR10)
def get_cifar10_dataset(root_path, img_size, add_inverse=False):
    img_size = 32 if img_size is None else img_size
    label_transform = None

    training_data = get_cifar10_train(
        root_path,
        img_size,
        add_inverse,
        label_transform,
    )
    test_data = get_cifar10_test(
        root_path,
        img_size,
        add_inverse,
        label_transform,
    )
    return training_data, test_data


def get_cifar10_train(root_path, img_size, add_inverse=False, label_transform=None):
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            (
                AddInverse()
                if add_inverse
                else torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ),
        ]
    )
    training_data = datasets.CIFAR10(
        root=root_path,
        train=True,
        download=False,
        transform=train_transform,
        target_transform=label_transform,
    )

    return training_data


def get_cifar10_test(root_path, img_size, add_inverse=False, label_transform=None):
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            (
                AddInverse()
                if add_inverse
                else torchvision.transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ),
        ]
    )
    test_data = datasets.CIFAR10(
        root=root_path,
        train=False,
        download=False,
        transform=test_transform,
        target_transform=label_transform,
    )

    return test_data


@register_dataset(DatasetSwitch.MNIST)
def get_mnist_dataset(root_path, img_size, **kwargs):
    img_size = 28 if img_size is None else img_size
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
        root=root_path,
        train=True,
        download=False,
        transform=transform,
        target_transform=label_transform,
    )

    test_data = datasets.MNIST(
        root=root_path,
        train=False,
        download=False,
        transform=transform,
        target_transform=label_transform,
    )

    return training_data, test_data
