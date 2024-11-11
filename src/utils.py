import argparse
from enum import Enum


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

class DatasetSwitch(ConvertableEnum):
    CIFAR10 = 200
    MNIST = 201