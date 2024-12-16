from src.models.resnet import BasicBlock, Bottleneck, ResNet
from src.models.simple_cnn import SimpleConvNet, SimpleConvSKBN
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
    if model_name in [
        ModelSwitch.SIMPLE_CNN,
        ModelSwitch.SIMPLE_CNN_BN,
        ModelSwitch.SIMPLE_CNN_SK,
        ModelSwitch.SIMPLE_CNN_SK_BN,
    ]:
        bn = model_name in [
            ModelSwitch.SIMPLE_CNN_BN,
            ModelSwitch.SIMPLE_CNN_SK_BN,
        ]
        sk = model_name in [
            ModelSwitch.SIMPLE_CNN_SK,
            ModelSwitch.SIMPLE_CNN_SK_BN,
        ]
        layers = kwargs.get("layers", None)
        return SimpleConvSKBN(
            input_shape,
            num_classes,
            activation_fn,
            conv_bias=bias,
            fc_bias=True,
            bn=bn,
            sk=sk,
            layers=layers,
        )

    if ModelSwitch.MNIST_CONV_NET == model_name:
        return SimpleConvNet(
            input_shape,
            num_classes,
            activation_fn,
            conv_bias=bias,
            fc_bias=True,
            layers=layers,
        )

    if ModelSwitch.RESNET_BASIC == model_name:
        assert "layers" in kwargs, "layers must be provided for ResNetBasic"
        layers = kwargs["layers"]
        return ResNet(
            BasicBlock,
            layers=layers,
            activation=activation_fn,
            num_classes=num_classes,
            input_shape=input_shape,
            conv_bias=bias,
            pre_act=pre_act,
            fc_bias=True,
        )

    if ModelSwitch.RESNET_BOTTLENECK == model_name:
        assert "layers" in kwargs, "layers must be provided for ResNetBottleneck"
        layers = kwargs["layers"]
        return ResNet(
            Bottleneck,
            layers=layers,
            activation=activation_fn,
            num_classes=num_classes,
            input_shape=input_shape,
            conv_bias=bias,
            pre_act=pre_act,
            fc_bias=True,
        )

    if ModelSwitch.RESNET18 == model_name:
        assert (
            "layers" not in kwargs
        ), "layers must not be provided for ResNet18, it seems to be an error"
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
        assert (
            "layers" not in kwargs
        ), "layers must not be provided for ResNet18, it seems to be an error"
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
        assert (
            "layers" not in kwargs
        ), "layers must not be provided for ResNet18, it seems to be an error"
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
