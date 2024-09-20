import torch.nn.functional as F
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_error_metrics(given_obs, generated_data):
    mse = F.mse_loss(given_obs, generated_data)
    mae = F.l1_loss(given_obs, generated_data)
    print("Mean Squared Error (MSE):", mse.item())
    print("Mean Absolute Error (MAE):", mae.item())

def ssim_index(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    # SSIM computation for each image in the batch over all timesteps
    ssim_scores = [ssim(y_true[i, t, 0], y_pred[i, t, 0], data_range=y_pred[i, t, 0].max() - y_pred[i, t, 0].min())
                   for i in range(y_true.shape[0]) for t in range(y_true.shape[1])]
    return np.mean(ssim_scores)

def calculate_reconstruction_error(original_data: torch.Tensor, reconstructed_data: torch.Tensor) -> dict:
    """
    计算重构误差，包括R², SSIM和NRMSE。

    Args:
        original_data (torch.Tensor): 原始输入数据，形状为 (样本数, 通道数, 高度, 宽度)。
        reconstructed_data (torch.Tensor): 重构后的数据，形状为 (样本数, 通道数, 高度, 宽度)。

    Returns:
        dict: 每个通道的重构误差，包括R², SSIM和NRMSE。
    """
    n_samples, n_channels, _, _ = original_data.shape
    errors = {}

    for channel in range(n_channels):
        # Flatten the images for R² and NRMSE calculation
        original_channel = original_data[:, channel].reshape(n_samples, -1)
        reconstructed_channel = reconstructed_data[:, channel].reshape(n_samples, -1)
        
        # Calculate R²
        mse = torch.mean((original_channel - reconstructed_channel) ** 2, dim=1)
        mean_original = torch.mean(original_channel, dim=1)
        ss_tot = torch.mean((original_channel - mean_original[:, None]) ** 2, dim=1)  # Adjusted dimension for broadcasting
        r2 = 1 - torch.mean(mse / ss_tot).item()
        errors[f'Channel {channel} R2'] = r2

        # Calculate SSIM
        original_channel_expanded = original_data[:, channel].unsqueeze(1)  # Expand to [batch_size, 1, height, width]
        reconstructed_channel_expanded = reconstructed_data[:, channel].unsqueeze(1)
        
        ssim_value = ssim_index(original_channel_expanded, reconstructed_channel_expanded)
        errors[f'Channel {channel} SSIM'] = ssim_value

        # Calculate NRMSE
        rmse = torch.sqrt(mse).mean().item()
        max_original = original_channel.max().item()
        min_original = original_channel.min().item()
        nrmse = rmse / (max_original - min_original)
        errors[f'Channel {channel} NRMSE'] = nrmse

    return errors