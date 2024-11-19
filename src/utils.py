import argparse
from enum import Enum

import torch
from torch import nn


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


class ActivationSwitch(ConvertableEnum):
    RELU = 10
    SOFTPLUS_B1 = 11
    SOFTPLUS_B10 = 12
    SOFTPLUS_B100 = 13
    SOFTPLUS_B1000 = 14
    SOFTPLUS_B10000 = 15
    SOFTPLUS_B100000 = 16
    LEAKY_RELU = 21


class DatasetSwitch(ConvertableEnum):
    CIFAR10 = 200
    MNIST = 201


def save_pth(model, path):
    """Saves the model to a .pth file.

    Args:
      model: The model to save.
      path: The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def convert_str_to_activation_fn(activation):
    if ActivationSwitch.RELU == activation:
        return nn.ReLU()
    if ActivationSwitch.LEAKY_RELU == activation:
        return nn.LeakyReLU()

    str_activation = str(activation)
    if "SOFTPLUS" in str_activation:
        beta = str_activation.replace("SOFTPLUS_B", "")
        beta = float(beta)

        return nn.Softplus(beta)

    raise NameError(str_activation)


def convert_str_to_loss_fn(loss):
    if LossSwitch.MSE == loss:
        return nn.MSELoss(reduce="sum")

    if LossSwitch.CE == loss:
        return nn.CrossEntropyLoss(reduce="sum")

    raise NameError(loss)
