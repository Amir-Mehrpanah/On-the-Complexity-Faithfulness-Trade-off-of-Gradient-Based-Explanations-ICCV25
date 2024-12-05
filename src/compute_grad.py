import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

from src.datasets import RepeatedBatchSampler, get_training_and_test_dataloader
from src.models import get_model
from src.utils import (
    ActivationSwitch,
    AugmentationSwitch,
    DatasetSwitch,
    LossSwitch,
    convert_str_to_activation_fn,
    convert_str_to_loss_fn,
    get_save_path,
)


def get_inputs():
    parser = argparse.ArgumentParser(description="Get inputs for the summary stats.")
    parser.add_argument(
        "--block_main",
        action="store_true",
        help="Block the main thread",
    )
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
    parser.add_argument(
        "--activation",
        type=ActivationSwitch.convert,
        required=True,
        help="Activation function (e.g., relu, softplus)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help="Image size",
    )
    parser.add_argument(
        "--augmentation",
        type=AugmentationSwitch.convert,
        required=True,
        help="use data augmentation",
    )
    parser.set_defaults(bias=True)
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_false",
        help="if the network has no bias",
    )
    parser.add_argument(
        "--add_inverse",
        action="store_true",
        help="add inverse to the dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for the dataloader",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="prefetch factor for the dataloader",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--eval_only_on_test",
        action="store_true",
        help="only evaluate on the test set",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="maximum number of batches to process",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="number of batches to to sample",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="which epoch to load",
    )

    args = parser.parse_args()
    args = vars(args)
    return args


def compute_input_grad(
    model,
    x,
    y,
):
    x.requires_grad_(True)
    output = model(x)
    output = output - output.logsumexp(-1)
    correct = (output.argmax(1) == y).sum().item()
    output = output.max()
    output.backward()
    grad = x.grad
    return grad, correct


async def async_torch_save(idx, obj=None):
    path = f"output_{idx}.pth"
    print(f"Saving {path} started.")
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, torch.save, obj, path)

    print(f"Saving {path} completed.")


async def compute_grad_and_save(test_dataloader, model, max_batches, device):
    tasks = []
    for i, (x, y) in enumerate(test_dataloader):
        x, y = x.to(device), y.to(device)

        grad, correct = compute_input_grad(model, x, y)
        tasks.append(
            asyncio.create_task(
                async_torch_save(
                    i,
                    obj={
                        "grad_mx2": torch.mean(grad**2, dim=0).detach().cpu(),
                        "grad_v": torch.var(grad, dim=0).detach().cpu(),
                        "correct": correct,
                    },
                )
            )
        )
        if max_batches > 0 and i >= max_batches:
            break

    await asyncio.gather(*tasks)  # Wait for all tasks to complete


def main(
    root_path,
    dataset,
    batch_size,
    img_size,
    augmentation,
    add_inverse,
    num_workers,
    prefetch_factor,
    model_name,
    activation,
    bias,
    epoch,
    eval_only_on_test,
    max_batches,
    num_batches,
    device,
    **args,
):
    activation_fn = convert_str_to_activation_fn(activation)
    sampler = (
        lambda datasource: RepeatedBatchSampler(
            datasource=datasource, num_repeats=batch_size * num_batches
        ),
    )
    aux = get_training_and_test_dataloader(
        dataset,
        root_path,
        batch_size,
        img_size=img_size,
        augmentation=augmentation,
        add_inverse=add_inverse,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        get_only_test=eval_only_on_test,
        shuffle=False,
        sampler=sampler,
    )

    if eval_only_on_test:
        test_dataloader, input_shape, num_classes = aux
    else:
        train_dataloader, test_dataloader, input_shape, num_classes = aux

    checkpoint_filename = get_save_path(
        model_name,
        activation,
        augmentation,
        bias,
        epoch,
        add_inverse,
    )
    model = get_model(
        input_shape=input_shape,
        model_name=model_name,
        num_classes=num_classes,
        activation_fn=activation_fn,
        bias=bias,
        add_inverse=add_inverse,
    ).to(device)

    checkpoint = torch.load(checkpoint_filename, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    asyncio.run(
        compute_grad_and_save(
            test_dataloader,
            model,
            max_batches,
            device,
        )
    )
    if eval_only_on_test:
        return

    # asyncio.run(
    #     compute_grad_and_save(
    #         train_dataloader,
    #         model,
    #         max_batches,
    #         device,
    #     )
    # )
