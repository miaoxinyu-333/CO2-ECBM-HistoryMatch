import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 定义输入门、遗忘门、输出门
        self.conv_i = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_f = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_o = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.conv_g = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        
    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)  # 拼接输入和隐藏状态
        
        i = torch.sigmoid(self.conv_i(combined))  # 输入门
        f = torch.sigmoid(self.conv_f(combined))  # 遗忘门
        o = torch.sigmoid(self.conv_o(combined))  # 输出门
        g = torch.tanh(self.conv_g(combined))    # 候选记忆
        
        c_next = f * c + i * g  # 更新记忆细胞
        h_next = o * torch.tanh(c_next)  # 更新隐藏状态
        
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, num_layers=1):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # 生成多个 ConvLSTM 单元的堆叠
        self.cells = nn.ModuleList([
            ConvLSTMCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])
        
        # 最终输出的卷积层，将 hidden_channels 转换为 output_channels（如 1 通道）
        self.conv_out = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        
        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device)
        
        outputs = []
        for t in range(time_steps):
            # 处理每个时间步的输入
            x_t = x[:, t]  # (batch_size, channels, height, width)
            
            # 逐层传递信息
            for i in range(self.num_layers):
                h, c = self.cells[i](x_t, h, c)
                
            # 将隐藏状态通过最后的卷积层，得到期望的输出通道数
            output_t = self.conv_out(h)  # (batch_size, output_channels, height, width)
            outputs.append(output_t.unsqueeze(1))  # 保存每个时间步的输出
            
        # 连接所有时间步的输出
        outputs = torch.cat(outputs, dim=1)  # (batch_size, time_steps, output_channels, height, width)
        
        return outputs