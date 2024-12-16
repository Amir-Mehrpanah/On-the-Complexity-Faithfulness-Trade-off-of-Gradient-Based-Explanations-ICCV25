from torch import nn


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        activation,
        conv_bias,
        fc_bias,
        layers,
        **kwargs,
    ):
        super().__init__()
        C, H, W = input_shape
        assert len(layers) == 1 and (
            layers[0] <= 4
        ), "layers must be a list of length 1 where layers[0] <= 4"

        self.features = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if layers[0] == 1:
            n_features = 32 * (H // 2) * (W // 2)
        elif layers[0] == 2:
            self.features.add_module(
                "conv2",
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            )
            self.features.add_module("activation2", activation)
            self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2))
            n_features = 64 * (H // 4) * (W // 4)
        elif layers[0] == 3:
            self.features.add_module(
                "conv2",
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            )
            self.features.add_module("activation2", activation)
            self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2))
            n_features = 128 * (H // 8) * (W // 8)
        elif layers[0] == 4:
            self.features.add_module(
                "conv2",
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            )
            self.features.add_module("activation2", activation)
            self.features.add_module("maxpool2", nn.MaxPool2d(kernel_size=2, stride=2))
            self.features.add_module(
                "conv3",
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=conv_bias),
            )
            self.features.add_module("activation3", activation)
            self.features.add_module("maxpool3", nn.MaxPool2d(kernel_size=2, stride=2))
            n_features = 256 * (H // 16) * (W // 16)

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128, bias=fc_bias),
            activation,
            nn.Linear(128, num_classes, bias=fc_bias),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SimpleConvSKBN(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        activation,
        conv_bias,
        fc_bias,
        bn,
        sk,
        layers,
    ):
        super().__init__()
        C, H, W = input_shape
        self.bn = bn
        self.sk = sk
        self.conv0 = nn.Conv2d(C, 32, 3, 1, 1, bias=conv_bias)
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1, bias=conv_bias)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=conv_bias)
        self.bn2 = nn.BatchNorm2d(64)
        self.downsample21 = nn.Conv2d(64, 32, 1, 1, 0, bias=conv_bias)
        self.conv3 = nn.Conv2d(32 * (2 - int(sk)), 64, 1, 1, 1, bias=conv_bias)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=conv_bias)
        self.bn4 = nn.BatchNorm2d(128)
        self.downsample43 = nn.Conv2d(128, 64, 1, 1, 0, bias=conv_bias)
        self.conv5 = nn.Conv2d(64 * (2 - int(sk)), 64, 1, 1, 1, bias=conv_bias)
        self.bn5 = nn.BatchNorm2d(64)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 15 * 15, 512, bias=fc_bias)
        self.fc2 = nn.Linear(512, num_classes, bias=fc_bias)

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(x)

        identity = x
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.activation(x)

        if self.sk:
            x = self.downsample21(x)
            x += identity

        x = self.maxpool(x)

        x = self.conv3(x)
        identity = x
        if self.bn:
            x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4(x)
        if self.bn:
            x = self.bn4(x)
        x = self.activation(x)

        if self.sk:
            x = self.downsample43(x)
            x += identity

        x = self.maxpool(x)

        x = self.conv5(x)
        if self.bn:
            x = self.bn5(x)
        x = self.activation(x)

        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
