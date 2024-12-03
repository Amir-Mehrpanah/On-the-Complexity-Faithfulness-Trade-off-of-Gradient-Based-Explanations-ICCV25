import argparse

from datasets import get_training_and_test_dataloader
from utils import DatasetSwitch


def get_inputs():
    parser = argparse.ArgumentParser(description="Get inputs for the summary stats.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen for debugger attach",
    )
    parser.add_argument(
        "--dataset",
        type=DatasetSwitch.convert,
        required=True,
        help="Dataset to use (e.g., cifar10, mnist)",
    )

    args = parser.parse_args()
    args = vars(args)
    return args


def main(
    root_path,
    dataset,
    batch_size,
    img_size,
    augmentation,
    add_inverse,
    num_workers,
    prefetch_factor,
    clip_quantile,
    **args,
):
    train_dataloader, test_dataloader, input_shape, num_classes = (
        get_training_and_test_dataloader(
            dataset,
            root_path,
            batch_size,
            img_size=img_size,
            augmentation=augmentation,
            add_inverse=add_inverse,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    )
