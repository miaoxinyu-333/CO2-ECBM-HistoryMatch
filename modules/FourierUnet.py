import torch
from torch import nn
from typing import List, Tuple, Union

from .activations import ACTIVATION_REGISTRY
from .blocks import FourierDownBlock, DownBlock, UpBlock, Upsample, MiddleBlock, Downsample

class FourierUnet(nn.Module):
    """Unet with Fourier layers in early downsampling blocks.

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input.
        time_future (int): Number of time steps in the output.
        hidden_channels (int): Number of channels in the first layer.
        activation (str): Activation function to use.
        modes1 (int): Number of Fourier modes to use in the first spatial dimension.
        modes2 (int): Number of Fourier modes to use in the second spatial dimension.
        norm (bool): Whether to use normalization.
        ch_mults (list): List of integers to multiply the number of channels by at each resolution.
        is_attn (list): List of booleans indicating whether to use attention at each resolution.
        mid_attn (bool): Whether to use attention in the middle block.
        n_blocks (int): Number of blocks to use at each resolution.
        n_fourier_layers (int): Number of early downsampling layers to use Fourier layers in.
        mode_scaling (bool): Whether to scale the number of modes with resolution.
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layer.
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
        modes1: int = 12,
        modes2: int = 12,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        n_fourier_layers: int = 2,
        mode_scaling: bool = True,
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
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    down.append(
                        FourierDownBlock(
                            in_channels,
                            out_channels,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            else:
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
        if use1x1:
            self.final = nn.Conv2d(n_channels, out_channels, kernel_size=1)
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
        return x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )