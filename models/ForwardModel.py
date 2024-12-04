import pytorch_lightning as pl
import torch

class ForwardModel(pl.LightningModule):
    def __init__(self, autoencoder_model, surrogate_model, device='cuda'):
        super(ForwardModel, self).__init__()

        # Assign the models to instance variables
        self.autoencoder_model = autoencoder_model
        self.surrogate_model = surrogate_model
        
        # Assign the device to an instance variable
        self.model_device = device
        
        # Move models to the specified device
        self.to(self.model_device)

        # Optional: Print or log model device information
        print(f"Models have been moved to device: {self.model_device}")

    def forward(self, x):
        # 将输入移动到指定设备
        x = x.to(self.model_device)  # shape (batch_size, 2, ...)

        # latents = self.autoencoder_model.encode(x).latent_dist.sample()
        # 使用 autoencoder 模型进行编码
        reconstructed = self.autoencoder_model.decode(x).sample

        reconstructed = reconstructed.unsqueeze(1)

        # 使用 surrogate 模型进行预测
        x_co2 = self.surrogate_model(reconstructed)

        return x_co2
