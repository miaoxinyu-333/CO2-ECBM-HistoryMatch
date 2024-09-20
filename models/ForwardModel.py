import pytorch_lightning as pl
import torch

class ForwardModel(pl.LightningModule):
    def __init__(self, pca_model_per, pca_model_por, autoencoder_per, autoencoder_por, surrogate_model, device='cuda'):
        super(ForwardModel, self).__init__()
        
        # Assign the models to instance variables
        self.pca_model_per = pca_model_per
        self.pca_model_por = pca_model_por
        self.autoencoder_per = autoencoder_per
        self.autoencoder_por = autoencoder_por
        self.surrogate_model = surrogate_model
        
        # Assign the device to an instance variable
        self.model_device = device
        
        # Move models to the specified device
        self.to(self.model_device)

        # Optional: Print or log model device information
        print(f"Models have been moved to device: {self.model_device}")

    def forward(self, x):
        # 将输入移动到指定设备
        x = x.to(self.model_device) # shape (batchsize,300)

        latent_per = x[:, 0, :]
        latent_por = x[:, 1, :]
        
        # 经过PCA逆变换
        x_pca_inverse_per = self.pca_model_per.inverse_transform(latent_per.cpu().numpy(), 1, 32, 32).to(self.model_device)  # shape :(batchsize,1,32,32)
        x_pca_inverse_por = self.pca_model_por.inverse_transform(latent_por.cpu().numpy(), 1, 32, 32).to(self.model_device)  # shape :(batchsize,1,32,32)

        x_pca_inverse_per = x_pca_inverse_per.unsqueeze(1)
        x_pca_inverse_por = x_pca_inverse_por.unsqueeze(1)

        x_autoencoder_per = self.autoencoder_per(x_pca_inverse_per)
        x_autoencoder_por = self.autoencoder_por(x_pca_inverse_por)

        x_combined = torch.cat((x_autoencoder_per, x_autoencoder_por), dim=2)

        x_co2 = self.surrogate_model(x_combined)

        return x_co2