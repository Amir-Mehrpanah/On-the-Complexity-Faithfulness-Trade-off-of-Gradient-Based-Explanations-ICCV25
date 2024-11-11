from torch import nn


class SimpleConvNet(nn.Module):
    def __init__(self, activation, conv_bias, fc_bias):
        super().__init__()
        self.activation = activation
        self.feature = nn.Sequential(
            nn.Conv2d(
                1,
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
                kernel_size=3,
                stride=1,
                padding=1,
                bias=conv_bias,
            ),
            self.activation,
            nn.Conv2d(
                128,
                out_channels=256,
                kernel_size=3,
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
                256 * 8 * 8,  # use 7x7 if 28x28 input
                512,
                bias=fc_bias,
            ),
            self.activation,
            nn.Linear(
                512,
                512,
                bias=fc_bias,
            ),
            self.activation,
            nn.Linear(
                512,
                2,
                bias=fc_bias,
            ),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
