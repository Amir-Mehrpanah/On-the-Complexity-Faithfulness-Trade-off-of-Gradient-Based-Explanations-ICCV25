import argparse
from enum import Enum
import os

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
    SIMPLE_CNN_DEPTH = 1000
    SIMPLE_CNN = 1001
    SIMPLE_CNN_BN = 1007
    SIMPLE_CNN_SK = 1008
    SIMPLE_CNN_SK_BN = 1009
    RESNET18 = 1002
    RESNET34 = 1003
    RESNET50 = 1004
    RESNET_BASIC = 1005
    RESNET_BOTTLENECK = 1006


class ExplainerSwitch(ConvertableEnum):
    # GRAD = 0 # implemented in our code
    GRAD_CAM = 1
    GUIDED_BPP = 2
    DEEP_LIFT = 3
    # SMOOTH_GRAD = 4 # implemented in our code
    INTEGRATED_GRAD = 5
    LRP = 6


class ActivationSwitch(ConvertableEnum):
    SOFTPLUS_B_01 = 11
    SOFTPLUS_B_1 = 12
    SOFTPLUS_B_2 = 13
    SOFTPLUS_B_3 = 30
    SOFTPLUS_B_4 = 31
    SOFTPLUS_B_5 = 24
    SOFTPLUS_B_6 = 26
    SOFTPLUS_B_7 = 27
    SOFTPLUS_B_8 = 28
    SOFTPLUS_B_9 = 25
    SOFTPLUS_B100 = 17
    SOFTPLUS_B10 = 16
    SOFTPLUS_B7 = 18
    SOFTPLUS_B50 = 20
    SOFTPLUS_B5 = 15
    SOFTPLUS_B3 = 19
    SOFTPLUS_B2 = 29
    SOFTPLUS_B1 = 14
    LEAKY_RELU = 21
    RELU = 10
    SIGMOID = 22
    TANH = 23


class DatasetSwitch(ConvertableEnum):
    CIFAR10 = 200
    MNIST = 201
    IMAGENETTE = 202
    FASHION_MNIST = 203
    IMAGENET = 204
    GRADS = 299


class AugmentationSwitch(ConvertableEnum):
    EXP_VIS = 1  # used in exaplanation methods for visualizing the original image
    EXP_GEN = 2  # used in exaplanation methods for generating the heatmaps
    TRAIN = 3  # used in training for training set


EXPERIMENT_PREFIX_SEP = "::"


def get_experiment_prefix(
    *,
    dataset,
    model_name,
    layers,
    activation,
    seed,
    l2_reg,
    img_size,
    lr,
    gaussian_noise_var,
    gaussian_blur_var,
    **args,
):
    name_list = []
    # name_list.append(dataset)
    name_list.append(model_name)
    name_list.append("_".join(map(str, layers)))
    name_list.append(activation)
    name_list.append(seed)
    name_list.append(l2_reg)
    name_list.append(gaussian_noise_var)
    name_list.append(gaussian_blur_var)
    name_list.append(lr)
    return os.path.join(
        str(dataset),
        str(img_size),
        EXPERIMENT_PREFIX_SEP.join(map(str, name_list)),
    )


def get_save_path(
    **kwargs,
):
    # augmentation = "aug" if augmentation else "noaug"
    # bias = "bias" if bias else "nobias"
    # add_inverse = "inv" if add_inverse else "noinv"
    experiment_prefix = get_experiment_prefix(**kwargs)
    return f"checkpoints/{experiment_prefix}.pt"


def save_pth(
    model,
    train_acc,
    test_acc,
    path,
):
    """Saves the model to a .pth file.

    Args:
      model: The model to save.
      path: The path to save the model to.
    """
    torch.save(
        {
            "model": model.state_dict(),
            "train_acc": train_acc,
            "test_acc": test_acc,
        },
        path,
    )


def convert_str_to_activation_fn(activation):
    str_activation = str(activation)

    if "LEAKY_RELU" in str_activation:
        return nn.LeakyReLU()

    if "RELU" in str_activation:
        return nn.ReLU()

    if "SIGMOID" in str_activation:
        return nn.Sigmoid()

    if "TANH" in str_activation:
        return nn.Tanh()

    if "SOFTPLUS" in str_activation:
        beta = str_activation.replace("SOFTPLUS_B", "")
        beta = beta.replace("_", ".")
        beta = float(beta)

        return nn.Softplus(beta)

    raise NameError(str_activation)


def convert_str_to_explainer(explainer, model, model_name):
    from captum import attr

    if ExplainerSwitch.GRAD_CAM == explainer:
        if ModelSwitch.RESNET18 == model_name:
            return attr.GuidedGradCam(model, model.layer4[1].conv2)
        if ModelSwitch.RESNET34 == model_name:
            return attr.GuidedGradCam(model, model.layer4[1].conv2)
        if ModelSwitch.RESNET50 == model_name:
            return attr.GuidedGradCam(model, model.layer4[2].conv3)
        if ModelSwitch.SIMPLE_CNN_DEPTH == model_name:
            return attr.GuidedGradCam(model, model.features[-1])
        raise NameError(model_name)

    if ExplainerSwitch.LRP == explainer:
        return attr.LRP(model)

    if ExplainerSwitch.DEEP_LIFT == explainer:
        return attr.DeepLift(model)
    
    if ExplainerSwitch.GUIDED_BPP == explainer:
        return attr.GuidedBackprop(model)
    
    if ExplainerSwitch.INTEGRATED_GRAD == explainer:
        return attr.IntegratedGradients(model)

    raise NameError(explainer)


def convert_str_to_loss_fn(loss):
    if LossSwitch.MSE == loss:
        return nn.MSELoss(reduction="sum")

    if LossSwitch.CE == loss:
        return nn.CrossEntropyLoss(reduction="sum")

    raise NameError(loss)
