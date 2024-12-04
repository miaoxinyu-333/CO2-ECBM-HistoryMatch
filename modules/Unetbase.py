"""
This file contains code derived from the open-source project by Microsoft Corporation.
The original code is available under the MIT License at https://github.com/pdearena/pdearena.git.

MIT License

Copyright (c) 2024 Xinyu Miao.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from .activations import ACTIVATION_REGISTRY

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu") -> None:
        super().__init__()
        self.activation = getattr(nn, activation.capitalize(), nn.ReLU)()  # Default to ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu") -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, activation)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Unetbase(nn.Module):
    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation="relu",
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.activation = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        insize = time_history * (n_input_scalar_components + n_input_vector_components * 2)
        outsize = time_future * (n_output_scalar_components + n_output_vector_components * 2)

        # Input projection
        self.image_proj = ConvBlock(insize, hidden_channels, activation)

        # Down-sampling layers (4 levels)
        self.down1 = Down(hidden_channels, hidden_channels * 2, activation)
        self.down2 = Down(hidden_channels * 2, hidden_channels * 4, activation)
        self.down3 = Down(hidden_channels * 4, hidden_channels * 8, activation)
        self.down4 = Down(hidden_channels * 8, hidden_channels * 16, activation)

        # Up-sampling layers (4 levels)
        self.up1 = Up(hidden_channels * 16, hidden_channels * 8, activation)
        self.up2 = Up(hidden_channels * 8, hidden_channels * 4, activation)
        self.up3 = Up(hidden_channels * 4, hidden_channels * 2, activation)
        self.up4 = Up(hidden_channels * 2, hidden_channels, activation)

        # Final output projection
        self.final = nn.Conv2d(hidden_channels, outsize, kernel_size=3, padding=1)

    def forward(self, x):
        assert x.dim() == 5  # Expecting [batch, time, channels, height, width]
        orig_shape = x.shape
        x = x.view(x.size(0), -1, *x.shape[3:])  # Flatten time and channels into one dimension

        # Down-sampling
        h1 = self.image_proj(x)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)

        # Up-sampling
        h = self.up1(h5, h4)
        h = self.up2(h, h3)
        h = self.up3(h, h2)
        h = self.up4(h, h1)

        # Final output
        x = self.final(h)
        return x.view(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
