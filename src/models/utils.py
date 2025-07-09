import os
import torch
from src.models.resnet import BasicBlock, Bottleneck, ResNet
from src.models.simple_cnn import SimpleConvNet, SimpleConvSKBN
from src.utils import ModelSwitch


def replace_gelu_with_relu(module: torch.nn.Module) -> None:
    """
    Recursively replaces all nn.GELU layers in ``module`` with nn.ReLU(inplace=True).
    The transformation is done in‑place and therefore mutates the original model.
    """
    for name, child in module.named_children():
        # If the child itself is a GELU, replace it
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.ReLU())
        else:
            # Otherwise recurse
            replace_gelu_with_relu(child)


def get_model(
    *,
    input_shape,
    model_name,
    num_classes,
    activation_fn,
    bias,
    pre_act,
    layers,
    checkpoint_path,
    device,
    **kwargs,
):
    model = None
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
        model = SimpleConvSKBN(
            input_shape,
            num_classes,
            activation_fn,
            conv_bias=bias,
            fc_bias=True,
            bn=bn,
            sk=sk,
        )

    if ModelSwitch.SIMPLE_CNN_DEPTH == model_name:
        model = SimpleConvNet(
            input_shape,
            num_classes,
            activation_fn,
            conv_bias=bias,
            fc_bias=True,
            layers=layers,
        )

    if ModelSwitch.RESNET_BASIC == model_name:
        model = ResNet(
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
        model = ResNet(
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
        model = ResNet(
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
        model = ResNet(
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
        model = ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            activation=activation_fn,
            num_classes=num_classes,
            input_shape=input_shape,
            conv_bias=bias,
            pre_act=pre_act,
            fc_bias=True,
        )

    if ModelSwitch.VIT_16 == model_name:
        import torchvision.models

        C, H, W = input_shape
        assert (H == 224) & (W == 224), "ViT-16 expects 224x224 input size."

        model = torchvision.models.vit_b_16()

        if isinstance(activation_fn, torch.nn.ReLU):
            print("Using ReLU activation function for ViT-16")
            replace_gelu_with_relu(model)
        elif isinstance(activation_fn, torch.nn.GELU):
            print("Using GELU, (default) activation function for ViT-16")
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    if ModelSwitch.VIT_32 == model_name:
        from torchvision.models import vit_b_32

        C, H, W = input_shape
        assert (H == 384) & (W == 384), "ViT-32 expects 384x384 input size."
        raise NotImplementedError("ViT-32 not implemented yet.")

    if model is None:
        raise NameError(model_name)

    model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        try:
            model.load_state_dict(checkpoint["model"])
        except KeyError:
            print("Loading model from older checkpoint")
            model.load_state_dict(checkpoint)  # for older checkpoints
    else:
        print("No checkpoint found")

    return model
