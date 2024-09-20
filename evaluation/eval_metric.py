import torch
from sklearn.metrics import r2_score
import numpy as np
from skimage.metrics import structural_similarity as ssim

def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def normalized_root_mean_squared_error(y_true, y_pred):
    y_true = y_true.view(-1)  # Flatten the tensor
    y_pred = y_pred.view(-1)  # Flatten the tensor
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    norm_factor = torch.mean(torch.abs(y_true))
    nrmse = rmse / norm_factor
    return nrmse

def r2_score_pytorch(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    # Flatten the tensors
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return r2_score(y_true, y_pred)

def ssim_index(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    # SSIM computation for each image in the batch over all timesteps
    ssim_scores = [ssim(y_true[i, t, 0], y_pred[i, t, 0], data_range=y_pred[i, t, 0].max() - y_pred[i, t, 0].min())
                   for i in range(y_true.shape[0]) for t in range(y_true.shape[1])]
    return np.mean(ssim_scores)