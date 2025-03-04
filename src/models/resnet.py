import copy
from typing import Any, Callable, List, Optional, Type, Union

from torch import nn
import torch


# copied from pytorch source code
def conv3x3(
    *,
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    conv_bias=False,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=conv_bias,
        dilation=dilation,
    )


# copied from pytorch source code
def conv1x1(
    *,
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    conv_bias=False,
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=conv_bias,
    )


# copied from pytorch source code
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        *,
        inplanes: int,
        planes: int,
        activation: Optional[Callable[..., nn.Module]],
        conv_bias,
        pre_act: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(
            in_planes=inplanes,
            out_planes=planes,
            stride=stride,
            conv_bias=conv_bias,
        )
        self.bn1 = norm_layer(inplanes if pre_act else planes)
        self.activation1 = copy.deepcopy(activation)
        self.activation2 = copy.deepcopy(activation)
        self.conv2 = conv3x3(
            in_planes=planes,
            out_planes=planes,
            conv_bias=conv_bias,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        if pre_act:
            self.forward = self.forward_preact
        else:
            self.forward = self.forward_

    def forward_preact(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bn1(x)
        out = self.activation1(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)

        out += identity

        return out

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation2(out)

        return out


# copied from pytorch source code
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        *,
        inplanes: int,
        planes: int,
        activation: Optional[Callable[..., nn.Module]],
        conv_bias,
        pre_act: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(
            in_planes=inplanes,
            out_planes=width,
            conv_bias=conv_bias,
        )
        self.bn1 = norm_layer(inplanes if pre_act else width)
        self.conv2 = conv3x3(
            in_planes=width,
            out_planes=width,
            stride=stride,
            groups=groups,
            dilation=dilation,
            conv_bias=conv_bias,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            in_planes=width,
            out_planes=planes * self.expansion,
            conv_bias=conv_bias,
        )
        self.bn3 = norm_layer(width if pre_act else planes * self.expansion)
        self.activation1 = copy.deepcopy(activation)
        self.activation2 = copy.deepcopy(activation)
        self.activation3 = copy.deepcopy(activation)
        self.downsample = downsample
        self.stride = stride

        if pre_act:
            self.forward = self.forward_preact
        else:
            self.forward = self.forward_

    def forward_preact(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.activation1(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation3(out)
        out = self.conv3(out)

        out += identity

        return out

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation3(out)

        return out


# copied from pytorch source code
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        input_shape,
        num_classes: int,
        conv_bias,
        fc_bias,
        activation: Optional[Callable[..., nn.Module]],
        pre_act: bool = False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        C, H, W = input_shape
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            C,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=conv_bias,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.activation1 = copy.deepcopy(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block=block,
            planes=64,
            blocks=layers[0],
            conv_bias=conv_bias,
            activation=activation,
            pre_act=pre_act,
        )
        self.layer2 = self._make_layer(
            block=block,
            planes=128,
            blocks=layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            conv_bias=conv_bias,
            activation=activation,
            pre_act=pre_act,
        )
        self.layer3 = self._make_layer(
            block=block,
            planes=256,
            blocks=layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            conv_bias=conv_bias,
            activation=activation,
            pre_act=pre_act,
        )
        self.layer4 = self._make_layer(
            block=block,
            planes=512,
            blocks=layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            conv_bias=conv_bias,
            activation=activation,
            pre_act=pre_act,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=fc_bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                assert isinstance(activation, nn.ReLU), "Code needs to be updated for other activations"
                # Relu
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                # general activation 
                # should use other init function that not depend on the activation function
                # nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        if pre_act:
            self.forward = self._forward_impl_preact
        else:
            self.forward = self._forward_impl

    def _make_layer(
        self,
        *,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        conv_bias,
        activation,
        pre_act: bool = False,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    in_planes=self.inplanes,
                    out_planes=planes * block.expansion,
                    stride=stride,
                    conv_bias=conv_bias,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                activation=activation,
                conv_bias=conv_bias,
                pre_act=pre_act,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation,
                    conv_bias=conv_bias,
                    pre_act=pre_act,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl_preact(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
