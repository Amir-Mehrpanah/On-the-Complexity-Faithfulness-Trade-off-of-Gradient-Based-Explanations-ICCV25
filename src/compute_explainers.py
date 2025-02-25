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
    convert_str_to_explainer,
    get_save_path,
)


def get_target_class(
    x,
    model,
):
    output = model(x)
    output = output - output.logsumexp(dim=-1, keepdim=True)
    target_label = output.argmax(-1).squeeze(0)
    return target_label


def compute_explainer_and_save(
    explainer,
    noisy_dataloader,
    model,
    num_distinct_images,
    num_batches,
    output_dir,
    stats,
    device,
    gaussian_noise_var,
):
    explanations = []
    print("Starting to compute grads")
    for i, (x, y) in enumerate(noisy_dataloader):
        x, y = x.to(device), y.to(device)
        target_class = get_target_class(x, model)
        correct = target_class == y
        if explainer.__class__.__name__ == "IntegratedGradients":
            baseline = torch.zeros_like(x)
            explanation = explainer.attribute(
                x, target=target_class, baselines=baseline
            )
        else:
            explanation = explainer.attribute(x, target=target_class)

        explanations.append(torch.mean(explanation, dim=0).detach().cpu())

        if (i + 1) % num_batches == 0:
            save_state(
                num_batches,
                output_dir,
                explanations,
                correct,
                i,
                x,
                y,
                gaussian_noise_var,
                stats,
            )

            explanations = []

        if (num_distinct_images > 0) and (i // num_batches >= num_distinct_images - 1):
            break


def rank_normalize(input_gradient):
    expected_shape = input_gradient.shape
    temp = rankdata(input_gradient.flatten()) / np.prod(expected_shape)
    return temp.reshape(expected_shape)


def save_state(
    num_batches,
    output_dir,
    explanations,
    correct,
    index,
    x,
    y,
    gaussian_noise_var,
    stats,
):
    explanations = torch.stack(explanations)
    assert len(explanations.shape) == 4

    if "mean" in stats or "mean_rank" in stats:
        agg_means = torch.mean(explanations, dim=0)

        if "mean" in stats:
            stats["mean"] = agg_means.sum(dim=0).detach().cpu()

        if "mean_rank" in stats:
            stats["mean_rank"] = rank_normalize(agg_means.sum(dim=0).numpy())

    if "image" in stats:
        stats["image"] = x[0].detach().cpu()

    if "label" in stats:
        stats["label"] = y[0].detach().cpu()

    if "batch_size" in stats:
        stats["batch_size"] = x.shape[0] * num_batches

    if "correct" in stats:
        stats["correct"] = correct.detach().cpu()

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
    explainer,
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
        gaussian_noise_var=0,
        gaussian_blur_var=0,
    )

    if eval_only_on_test:
        exp_gen_test_dataloader, input_shape, num_classes = exp_gen_loaders
    else:
        exp_gen_train_dataloader, exp_gen_test_dataloader, input_shape, num_classes = (
            exp_gen_loaders
        )

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

    explainer = convert_str_to_explainer(explainer, model, model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint["model"])
    except KeyError:
        print("Loading model from older checkpoint")
        model.load_state_dict(checkpoint)  # for older checkpoints

    model.eval()

    compute_explainer_and_save(
        explainer,
        exp_gen_test_dataloader,
        model,
        num_distinct_images,
        num_batches,
        output_dir,
        stats,
        device,
        0,
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
