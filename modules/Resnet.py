import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, norm: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(3, 3, 3), stride=(stride, 1, 1), padding=(1, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(planes) if norm else nn.Identity()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.bn2 = nn.BatchNorm3d(planes) if norm else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1, 1), bias=False),
                nn.BatchNorm3d(self.expansion * planes) if norm else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return out

class ResNet3D(nn.Module):
    def __init__(
        self, 
        n_input_channels: int, 
        n_output_channels: int, 
        num_blocks: list, 
        hidden_channels: int = 64,
        activation: str = "relu",
        norm: bool = True
    ):
        super().__init__()
        self.in_planes = hidden_channels

        # Input layer
        self.conv_in = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=True)
        
        # Create the ResNet layers
        self.layers = nn.ModuleList([
            self._make_layer(BasicBlock, self.in_planes, num_blocks[i], stride=1, norm=norm) 
            for i in range(len(num_blocks))
        ])
        
        # Output layer
        self.conv_out = nn.Conv3d(self.in_planes, n_output_channels, kernel_size=(3, 3, 3), stride=1, padding=1)

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int, norm: bool):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_planes, planes, stride=stride, norm=norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_out(x)
        return x

# Example usage:
# n_input_channels: 2 (time steps), n_output_channels: 12 (channels for each output)
# num_blocks: A list specifying how many basic blocks for each layer
model = ResNet3D(n_input_channels=2, n_output_channels=12, num_blocks=[2, 2, 2], hidden_channels=64)

# Test with a dummy input matching your input shape: [batch_size, time_steps, channels, height, width]
dummy_input = torch.randn(4983, 2, 1, 64, 64)  # batch_size=4983, time_steps=2, channels=1, 64x64 resolution
output = model(dummy_input)

print(f"Output shape: {output.shape}")  # Expected output shape: [batch_size, n_output_channels, time_steps, height, width]
