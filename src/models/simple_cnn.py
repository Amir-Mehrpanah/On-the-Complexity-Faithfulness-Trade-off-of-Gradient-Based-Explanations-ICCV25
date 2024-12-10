from torch import nn


class SimpleConvNet(nn.Module):
    def __init__(self, input_shape, num_classes, activation, conv_bias, fc_bias):
        super().__init__()
        C, H, W = input_shape
        self.activation = activation
        self.feature = nn.Sequential(
            nn.Conv2d(
                C,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
            self.activation,
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(
                128 * 8 * 8,  # use 7x7 if 28x28 input, 4x4 if 32x32 input
                512,
                bias=fc_bias,
            ),
            self.activation,
            nn.Linear(
                512,
                num_classes,
                bias=fc_bias,
            ),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
