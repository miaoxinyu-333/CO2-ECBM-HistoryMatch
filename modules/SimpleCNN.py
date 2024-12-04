import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, time_steps=12):
        super(SimpleCNN, self).__init__()
        self.time_steps = time_steps  # 输出的时间步数
        
        # 卷积层：用于提取空间特征
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size, _, channels, height, width = x.size()
        
        # 检查输入时间步数是否为 1
        assert x.size(1) == 1, "Input must have exactly 1 time step"
        
        # 提取单时间步的数据
        x_t = x[:, 0]  # 形状 (batch_size, channels, height, width)
        
        # 通过卷积层提取空间特征
        x_t = self.relu(self.conv1(x_t))
        x_t = self.relu(self.conv2(x_t))
        x_t = self.conv3(x_t)  # 形状 (batch_size, output_channels, height, width)
        
        # 将单个时间步扩展为多个时间步
        outputs = x_t.unsqueeze(1).repeat(1, self.time_steps, 1, 1, 1)  # (batch_size, time_steps, output_channels, height, width)
        
        return outputs
