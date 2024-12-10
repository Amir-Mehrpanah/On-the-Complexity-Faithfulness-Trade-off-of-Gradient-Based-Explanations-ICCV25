from src.models.resnet import BasicBlock, Bottleneck, ResNet
from src.models.simple_cnn import SimpleConvNet
from src.utils import ModelSwitch


def get_model(
    *,
    input_shape,
    model_name,
    num_classes,
    activation_fn,
    bias,
    pre_act,
    **kwargs,
):
    if ModelSwitch.SIMPLE_CNN == model_name:
        return SimpleConvNet(
            input_shape,
            num_classes,
            activation_fn,
            bias,
            True,
        )

    if ModelSwitch.RESNET18 == model_name:
        return ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            activation=activation_fn,
            num_classes=num_classes,
            input_shape=input_shape,
            conv_bias=bias,
            pre_act=pre_act,
            fc_bias=True,
        )

    if ModelSwitch.RESNET34 == model_name:
        return ResNet(
            BasicBlock,
            [3, 4, 6, 3],
            activation=activation_fn,
            num_classes=num_classes,
            input_shape=input_shape,
            conv_bias=bias,
            pre_act=pre_act,
            fc_bias=True,
        )

    if ModelSwitch.RESNET50 == model_name:
        return ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            activation=activation_fn,
            num_classes=num_classes,
            input_shape=input_shape,
            conv_bias=bias,
            pre_act=pre_act,
            fc_bias=True,
        )

    raise NameError(model_name)
