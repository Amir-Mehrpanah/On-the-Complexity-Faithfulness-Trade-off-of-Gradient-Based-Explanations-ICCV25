import argparse
from enum import Enum

import torch
from torch import nn


def determine_device(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu" if args["port"] == 0 else device
    args["device"] = device
    print(f"Using {device} device")


class ConvertableEnum(Enum):
    @classmethod
    def convert(cls, value):
        try:
            return cls[value]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Invalid {cls.__name__} value '{value}'. Choose from {', '.join([e.name for e in cls])}."
            )

    def __str__(self):
        return self.name


class LossSwitch(ConvertableEnum):
    MSE = 0
    CE = 1


class ModelSwitch(ConvertableEnum):
    SIMPLE_CNN = 1001
    RESNET18 = 1002
    RESNET34 = 1003
    RESNET50 = 1004
    RESNET_BASIC = 1005
    RESNET_BOTTLENECK = 1006


class ActivationSwitch(ConvertableEnum):
    SOFTPLUS_B_01 = 11
    SOFTPLUS_B_1 = 12
    SOFTPLUS_B_2 = 13
    SOFTPLUS_B100 = 17
    SOFTPLUS_B10 = 16
    SOFTPLUS_B5 = 15
    SOFTPLUS_B1 = 14
    LEAKY_RELU = 21
    RELU = 10


class DatasetSwitch(ConvertableEnum):
    CIFAR10 = 200
    MNIST = 201
    IMAGENETTE = 202


class AugmentationSwitch(ConvertableEnum):
    EXP_VIS = 1  # used in exaplanation methods for visualizing the original image
    EXP_GEN = 2  # used in exaplanation methods for generating the heatmaps
    TRAIN = 3  # used in training for training set


def get_experiment_prefix(
    model_name,
    activation,
    augmentation,
    bias,
    epoch,
    add_inverse,
    pre_act,
    layers,
):
    return f"{model_name}_{layers}"


def get_save_path(
    **kwargs,
):
    # augmentation = "aug" if augmentation else "noaug"
    # bias = "bias" if bias else "nobias"
    # add_inverse = "inv" if add_inverse else "noinv"
    experiment_prefix = get_experiment_prefix(
        **kwargs
    )
    return f"checkpoints/{experiment_prefix}.pt"


def save_pth(model, path):
    """Saves the model to a .pth file.

    Args:
      model: The model to save.
      path: The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def convert_str_to_activation_fn(activation):
    str_activation = str(activation)
    if "LEAKY_RELU" in str_activation:
        return nn.LeakyReLU()
    if "RELU" in str_activation:
        return nn.ReLU()

    if "SOFTPLUS" in str_activation:
        beta = str_activation.replace("SOFTPLUS_B", "")
        beta = beta.replace("_", ".")
        beta = float(beta)

        return nn.Softplus(beta)

    raise NameError(str_activation)


def convert_str_to_loss_fn(loss):
    if LossSwitch.MSE == loss:
        return nn.MSELoss(reduction="sum")

    if LossSwitch.CE == loss:
        return nn.CrossEntropyLoss(reduction="sum")

    raise NameError(loss)
