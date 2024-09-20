import torch
from torch import nn
from typing import List, Tuple, Union

from .activations import ACTIVATION_REGISTRY
from .blocks import DownBlock, UpBlock, Upsample, MiddleBlock, Downsample

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


class Unet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input
        time_future (int): Number of time steps in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation: str,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * 2)
        n_channels = hidden_channels
        # Project image into feature map
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, has_attn=mid_attn, activation=activation, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * 2)
        #
        if use1x1:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5
        orig_shape = x.shape
        x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        x = x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
        return x
