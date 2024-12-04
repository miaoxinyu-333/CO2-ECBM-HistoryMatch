import sys
import os
import torch
from torch.utils.data import random_split
import pickle

def normalized_root_mean_squared_error(original_tensor, reconstructed_tensor):
    """
    计算 Normalized Root Mean Squared Error (NRMSE)
    :param original_tensor: 原始张量
    :param reconstructed_tensor: 重构张量
    :return: NRMSE 值
    """
    mse = torch.mean((original_tensor - reconstructed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    range_val = torch.max(original_tensor) - torch.min(original_tensor)
    nrmse = rmse / (range_val + 1e-8)  # 防止除以零
    return nrmse.item()

def mean_absolute_error(original_tensor, reconstructed_tensor):
    """计算 Mean Absolute Error (MAE)"""
    mae = torch.mean(torch.abs(original_tensor - reconstructed_tensor))
    return mae.item()


def root_mean_squared_error(original_tensor, reconstructed_tensor):
    mse = torch.mean((original_tensor - reconstructed_tensor) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

# 替换 NRMSE 为 MAPE
def mean_absolute_percentage_error(original_tensor, reconstructed_tensor):
    epsilon = 1e-8  # 防止除零
    percentage_error = torch.abs((original_tensor - reconstructed_tensor) / (original_tensor + epsilon))
    mape = torch.mean(percentage_error) * 100  # 转为百分比
    return mape.item()

# 获取项目根目录
def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def save_pca_model(pca_model, save_dir="logs/PCA_PKL"):
    """Save the trained PCA model as a .pkl file."""
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the file name
    file_name = os.path.join(save_dir, "pca_model.pkl")

    # Save the PCA model
    with open(file_name, "wb") as f:
        pickle.dump(pca_model, f)
    print(f"PCA model saved to {file_name}")

def main():
    # 设置项目根目录
    setup_project_root()
    
    # 导入所需模块
    from evaluation.eval_metric import ssim_index
    from evaluation.eval_metric import r2_score_pytorch
    from models.PCAModel import PCAModel
    from utils.dataUtils import load_per_por_from_h5

    # 加载数据
    data_set_path = "dataSet/surrogate/dataset_per_por_4983.h5"
    per_tensor, por_tensor = load_per_por_from_h5(data_set_path)

    # 调整数据形状
    per_tensor = per_tensor.unsqueeze(1)
    por_tensor = por_tensor.unsqueeze(1)
    per_tensor = torch.log(per_tensor + 1e-8)  # 防止 log(0) 的情况
    per_min, per_max = per_tensor.min(), per_tensor.max()
    per_tensor = (per_tensor - per_min) / (per_max - per_min)  # 归一化到 [0, 1]
    por_min, por_max = por_tensor.min(), por_tensor.max()
    por_tensor = (por_tensor - por_min) / (por_max - por_min)  # 归一化到 [0, 1]
    inputs_tensor = torch.cat((per_tensor, por_tensor), dim=1)

    print(f"Original input shape: {inputs_tensor.shape}")  # torch.Size([4983, 2, 64, 64])

    # 数据集划分
    train_size = int(0.7 * inputs_tensor.size(0))  # 70% 训练集
    test_size = inputs_tensor.size(0) - train_size  # 30% 测试集
    train_data, test_data = random_split(inputs_tensor, [train_size, test_size])
    
    train_data = torch.stack(list(train_data))  # 转换为张量
    test_data = torch.stack(list(test_data))

    print(f"Train data shape: {train_data.shape}")  # e.g., torch.Size([3986, 2, 64, 64])
    print(f"Test data shape: {test_data.shape}")  # e.g., torch.Size([997, 2, 64, 64])

    # 定义 PCA 模型
    latent_dim = 256
    pca_model = PCAModel(n_components=latent_dim)

    # 在训练集上进行 PCA
    flattened_train = train_data.view(train_data.size(0), -1).numpy()  # 转为 numpy
    pca_model.fit(flattened_train)

    # 保存 PCA 模型
    save_pca_model(pca_model)

    # 降维与反PCA测试
    flattened_test = test_data.view(test_data.size(0), -1).numpy()
    reduced_test = pca_model.transform(flattened_test)
    reconstructed_test = pca_model.inverse_transform(
        reduced_test, 
        n_channels=test_data.size(1), 
        height=test_data.size(2), 
        width=test_data.size(3)
    )

    # 计算指标
    ssim = ssim_index(test_data, reconstructed_test)
    print(f"SSIM Index on Test Data: {ssim:.6f}")

    r2 = r2_score_pytorch(test_data, reconstructed_test)
    print(f"R² Score on Test Data: {r2:.6f}")

    # 示例：使用 RMSE 和 MAPE
    rmse = root_mean_squared_error(test_data, reconstructed_test)
    print(f"RMSE: {rmse:.6f}")

    nrmse = normalized_root_mean_squared_error(test_data, reconstructed_test)
    print(f"NRMSE: {nrmse:.6f}")

    mape = mean_absolute_percentage_error(test_data, reconstructed_test)
    print(f"MAPE: {mape:.6f}")

    mae = mean_absolute_error(test_data, reconstructed_test)
    print(f"MAE: {mae:.6f}")

if __name__ == "__main__":
    main()
