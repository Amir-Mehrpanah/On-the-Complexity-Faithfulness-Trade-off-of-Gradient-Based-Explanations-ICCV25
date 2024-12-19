import argparse
import numpy as np
from scipy.stats import rankdata
import os
import torch

from src.datasets import RepeatedSequentialSampler, get_training_and_test_dataloader
from src.models.utils import get_model
from src.utils import (
    ActivationSwitch,
    AugmentationSwitch,
    DatasetSwitch,
    ModelSwitch,
    convert_str_to_activation_fn,
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
        type=ModelSwitch.convert,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--eval_only_on_test",
        action="store_true",
        help="only evaluate on the test set",
    )
    parser.add_argument(
        "--num_distinct_images",
        type=int,
        default=-1,
        help="maximum number of batches to process enter -1 to process all",
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
    parser.add_argument(
        "--gaussian_noise_var",
        type=float,
        default=1e-5,
        help="variance of the gaussian noise",
    )
    parser.add_argument(
        "--pre_act",
        action="store_true",
        help="use preact architecture",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="number of layers",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="timeout for the job",
    )

    # this variable more likely to be a constant. unnecessary to be passed as an argument
    stats = {
        "mean_rank": None,
        "var_rank": None,
        "mean": None,
        "var": None,
        "correct": None,
        "image": None,
        "label": None,
        "batch_size": None,
    }

    args = parser.parse_args()
    args = vars(args)
    args["stats"] = stats
    return args


def forward_single(
    x,
    model,
    target_class,
):
    output = model(x)
    output = output - output.logsumexp(dim=-1, keepdim=True)
    return output[:, target_class].squeeze(0), output


def get_target_class(
    x,
    model,
):
    output = model(x)
    output = output - output.logsumexp(dim=-1, keepdim=True)
    target_label = output.argmax(-1).squeeze(0)
    return target_label


def forward_single_grad(model, x, target_class):
    # assert that we have only one image
    assert x.ndim == 3, "x should have shape (C, H, W)"
    x = x.unsqueeze(0)  # (1, C, H, W)
    grad, output = torch.func.grad(forward_single, has_aux=True)(x, model, target_class)
    grad = grad.squeeze(0)  # (C, H, W)
    output = output.squeeze(0)  # (num_classes,)
    return grad, output


def forward_batch_grad(model, x, target_class):
    return torch.func.vmap(
        forward_single_grad,
        in_dims=(None, 0),
        out_dims=0,
    )(
        model,
        x,
        target_class,
    )


def compute_grad_and_save(
    clean_dataloader,
    noisy_dataloader,
    model,
    num_distinct_images,
    num_batches,
    output_dir,
    stats,
    device,
    gaussian_noise_var,
):
    grad_means = []
    grad_vars = []
    corrects = []
    iter_loader = iter(clean_dataloader)
    print("Starting to compute grads")
    for i, (x, y) in enumerate(noisy_dataloader):
        x, y = x.to(device), y.to(device)

        if i % num_batches == 0:
            x_clean, y_clean = next(iter_loader)
            target_class = get_target_class(x_clean, model)

        grad, output = forward_batch_grad(model, x, target_class)
        grad = grad**2

        correct = ((output.argmax(-1) == y).sum() / y.shape[0]).detach().cpu()
        grad_mean = torch.mean(grad, dim=0).detach().cpu()
        grad_var = torch.var(grad, dim=0).detach().cpu()

        grad_means.append(grad_mean)
        grad_vars.append(grad_var)
        corrects.append(correct)

        if (i + 1) % num_batches == 0:
            save_state(
                num_batches,
                output_dir,
                grad_means,
                grad_vars,
                corrects,
                i,
                x,
                y,
                gaussian_noise_var,
                stats,
            )

            grad_means = []
            grad_vars = []
            corrects = []

        if num_distinct_images > 0 and i // num_batches >= num_distinct_images:
            break


def rank_normalize(input_gradient):
    expected_shape = input_gradient.shape
    temp = rankdata(input_gradient.flatten()) / np.prod(expected_shape)
    return temp.reshape(expected_shape)


def save_state(
    num_batches,
    output_dir,
    grad_means,
    grad_vars,
    corrects,
    index,
    x,
    y,
    gaussian_noise_var,
    stats,
):
    grad_means = torch.stack(grad_means)

    if "mean" in stats or "mean_rank" in stats:
        agg_means = torch.mean(grad_means, dim=0)

        if "mean" in stats:
            stats["mean"] = agg_means.sum(dim=0).detach().cpu()

        if "mean_rank" in stats:
            stats["mean_rank"] = rank_normalize(agg_means.sum(dim=0).numpy())

    if "var" in stats or "var_rank" in stats:
        agg_vars = torch.mean(torch.stack(grad_vars), dim=0) + 1 / (
            max(len(grad_vars) - 1, 1)  # avoid division by zero in case of single batch
        ) * torch.sum((grad_means - agg_means) ** 2, dim=0)

        if "var" in stats:
            stats["var"] = agg_vars.sum(dim=0).detach().cpu()

        if "var_rank" in stats:
            agg_vars = rank_normalize(agg_vars.sum(dim=0).numpy())
            stats["var_rank"] = agg_vars

    if "correct" in stats:
        stats["correct"] = np.mean(corrects)

    if "image" in stats:
        stats["image"] = x[0].detach().cpu()

    if "label" in stats:
        stats["label"] = y[0].detach().cpu()

    if "batch_size" in stats:
        stats["batch_size"] = x.shape[0] * num_batches

    stats["noise_scale"] = gaussian_noise_var
    stats["index"] = index // num_batches
    torch.save(
        stats,
        os.path.join(output_dir, f"{index // num_batches}.pt"),
    )


def main(
    *,
    root_path,
    output_dir,
    dataset,
    batch_size,
    img_size,
    add_inverse,
    num_workers,
    prefetch_factor,
    model_name,
    activation,
    bias,
    eval_only_on_test,
    num_distinct_images,
    num_batches,
    gaussian_noise_var,
    stats,
    pre_act,
    layers,
    device,
    checkpoint_path,
    **args,
):
    print(locals())

    activation_fn = convert_str_to_activation_fn(activation)
    sampler = lambda datasource: RepeatedSequentialSampler(
        datasource=datasource,
        num_repeats=batch_size * num_batches,
    )
    exp_gen_loaders = get_training_and_test_dataloader(
        dataset,
        root_path,
        batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        get_only_test=eval_only_on_test,
        shuffle=False,
        sampler=sampler,
        img_size=img_size,
        augmentation=AugmentationSwitch.EXP_GEN,
        add_inverse=add_inverse,
        gaussian_noise_var=gaussian_noise_var,
    )
    clean_gen_loaders = get_training_and_test_dataloader(
        dataset,
        root_path,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        get_only_test=eval_only_on_test,
        shuffle=False,
        img_size=img_size,
        augmentation=AugmentationSwitch.EXP_GEN,
        add_inverse=add_inverse,
        gaussian_noise_var=0,
    )

    if eval_only_on_test:
        exp_gen_test_dataloader, input_shape, num_classes = exp_gen_loaders
        test_dataloader, _, _ = clean_gen_loaders
    else:
        exp_gen_train_dataloader, exp_gen_test_dataloader, input_shape, num_classes = (
            exp_gen_loaders
        )
        train_dataloader, test_dataloader, _, _ = clean_gen_loaders

    model = get_model(
        input_shape=input_shape,
        model_name=model_name,
        num_classes=num_classes,
        activation_fn=activation_fn,
        bias=bias,
        add_inverse=add_inverse,
        pre_act=pre_act,
        layers=layers,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    compute_grad_and_save(
        test_dataloader,
        exp_gen_test_dataloader,
        model,
        num_distinct_images,
        num_batches,
        output_dir,
        stats,
        device,
        gaussian_noise_var,
    )
    if eval_only_on_test:
        return

    #     compute_grad_and_save(
    #         train_dataloader,
    #         model,
    #         num_distinct_images,
    #         num_batches,
    #         output_dir,
    #         device,
    #     )
