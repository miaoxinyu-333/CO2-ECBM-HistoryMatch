import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
from utils.dataUtils import remove_prefix
from torch import nn, optim
from evaluation.eval_metric import (
    root_mean_squared_error,
    normalized_root_mean_squared_error,
    r2_score_pytorch,
    ssim_index,
    mean_absolute_error,
    mean_squared_error,
)

class AutoencoderKLWrapper(pl.LightningModule):
    def __init__(self, lr=1e-3, kl_weight=0.1, weight_decay=0.0):
        """
        初始化 AutoencoderKLWrapper

        Args:
            lr (float): 学习率
            kl_weight (float): KL 散度的权重
            weight_decay (float): 权重衰减（L2 正则化）
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 加载预训练模型
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        self.autoencoder = AutoencoderKL.from_single_file(url)
        
        # 修改模型配置
        self.autoencoder.config.in_channels = 2
        self.autoencoder.config.out_channels = 2
        self.autoencoder.config.sample_size = 64
        self.autoencoder.encoder.conv_in = nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1)
        self.autoencoder.decoder.conv_out = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        
        # 损失函数
        self.reconstruction_criterion = nn.MSELoss()
        self.kl_weight = kl_weight  # KL 散度损失的权重

    def configure_loss_function(self, reconstructed, original, latents):
        """
        配置自定义的损失函数，包括重构损失 (MSE) 和 KL 散度损失。
        """
        # 计算重构损失
        reconstruction_loss = self.reconstruction_criterion(reconstructed, original)

        # 提取 KL 散度相关参数
        latent_dist = latents.latent_dist
        mean, logvar = latent_dist.mean, latent_dist.logvar  # 使用 `logvar` 而不是 `log_var`

        # 计算 KL 散度
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        kl_divergence = kl_divergence.mean()

        # 总损失 = 重构损失 + KL 散度损失
        total_loss = reconstruction_loss + self.kl_weight * kl_divergence
        return total_loss, reconstruction_loss, kl_divergence

    def forward(self, x):
        latents = self.autoencoder.encode(x).latent_dist.sample()
        reconstructed = self.autoencoder.decode(latents).sample
        return reconstructed

    def log_metrics(self, y_true, y_pred, stage="train"):
        """
        记录评价指标
        """
        rmse = root_mean_squared_error(y_true, y_pred)
        # 数据提前已经进行了严格的[0,1]归一化 这里的rmse和nrmse是一样的 原有的nrmse函数适用于大范围 不再适用于这里的情况
        #nrmse = normalized_root_mean_squared_error(y_true, y_pred)
        r2 = r2_score_pytorch(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        ssim_value = ssim_index(y_true, y_pred)

        self.log(f"{stage}_rmse", rmse, prog_bar=True)
        #self.log(f"{stage}_nrmse", nrmse, prog_bar=True)
        self.log(f"{stage}_r2", r2, prog_bar=True)
        self.log(f"{stage}_mae", mae, prog_bar=True)
        self.log(f"{stage}_mse", mse, prog_bar=True)
        self.log(f"{stage}_ssim", ssim_value, prog_bar=True)

    def training_step(self, batch, batch_idx):
        # 解包 batch
        inputs, targets = batch

        # 编码和解码
        latents = self.autoencoder.encode(inputs)
        latent_samples = latents.latent_dist.sample()
        reconstructed = self.autoencoder.decode(latent_samples).sample

        # 计算总损失
        total_loss, reconstruction_loss, kl_divergence = self.configure_loss_function(reconstructed, targets, latents)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_reconstruction_loss", reconstruction_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_kl_divergence", kl_divergence, prog_bar=True, on_step=True, on_epoch=True)

        # 记录指标
        self.log_metrics(targets, reconstructed, stage="train")

        return total_loss

    def validation_step(self, batch, batch_idx):
        # 解包 batch
        inputs, targets = batch

        # 编码和解码
        latents = self.autoencoder.encode(inputs)
        latent_samples = latents.latent_dist.sample()
        reconstructed = self.autoencoder.decode(latent_samples).sample

        # 计算总损失
        total_loss, reconstruction_loss, kl_divergence = self.configure_loss_function(reconstructed, targets, latents)
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("val_kl_divergence", kl_divergence, prog_bar=True)

        # 记录指标
        self.log_metrics(targets, reconstructed, stage="val")

        return total_loss

    def test_step(self, batch, batch_idx):
        # 解包 batch
        inputs, targets = batch

        # 编码和解码
        latents = self.autoencoder.encode(inputs)
        latent_samples = latents.latent_dist.sample()
        reconstructed = self.autoencoder.decode(latent_samples).sample

        # 计算总损失
        total_loss, reconstruction_loss, kl_divergence = self.configure_loss_function(reconstructed, targets, latents)
        self.log("test_loss", total_loss, prog_bar=True)
        self.log("test_reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("test_kl_divergence", kl_divergence, prog_bar=True)

        # 记录指标
        self.log_metrics(targets, reconstructed, stage="test")

        return total_loss

    def configure_optimizers(self):
        """
        配置优化器，增加 weight_decay 参数。
        """
        optimizer = optim.Adam(
            self.autoencoder.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def load_model_weights(self, checkpoint_path):
        """加载训练好的模型权重，去掉前缀"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 如果 checkpoint 中有 "state_dict"，获取它
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # 去掉前缀 "autoencoder." 
        prefix = 'autoencoder.'
        checkpoint = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}

        # 加载去掉前缀后的权重
        self.autoencoder.load_state_dict(checkpoint)
        print(f"Loaded model weights from {checkpoint_path}")