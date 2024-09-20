import os
import h5py
import torch

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    归一化张量到 [0, 1] 区间。

    Args:
        tensor (torch.Tensor): 要归一化的张量。

    Returns:
        torch.Tensor: 归一化后的张量。
    """
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def load_datasets_from_h5(directory):
    """
    从指定目录的 HDF5 文件加载数据集，并以文件名作为变量名存储在字典中。
    """
    data_dict = {}
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            dataset_name = os.path.splitext(filename)[0]  # 去除扩展名，用作键名
            
            with h5py.File(file_path, 'r') as hdf:
                # 假设每个文件中都只有一个名为 'data' 的数据集
                data = hdf['data'][:]
                data_dict[dataset_name] = data

    return data_dict

def load_reconstuction_h5(h5_file_path):
    with h5py.File(h5_file_path, 'r') as h5f:
            original = h5f['original'][:]
            reconstructed = h5f['reconstructed'][:]
        
    original = torch.tensor(original, dtype=torch.float32)
    original = original.unsqueeze(1)
    reconstructed = torch.tensor(reconstructed, dtype=torch.float32)
    reconstructed = reconstructed.unsqueeze(1)


    return reconstructed, original


def observation_operator(model_output):
    # 提取下标为 (1, 1) 的数据
    return model_output[:, 0, 1, 1]

# 创建合成观测数据
def create_synthetic_obs(prior_data, i, j):
    # 提取所有样本中 (i, j) 点的值
    obs_data = prior_data[:, :, :, i, j].clone().detach().numpy()
    # 对所有样本取均值，得到单个观测数据
    single_obs = obs_data.mean(axis=0)
    return single_obs

def remove_prefix(state_dict, prefix):
    '''Old style model is stored with all names prefixed with `prefix`.'''
    n = len(prefix)
    return {k[n:]: v for k, v in state_dict.items() if k.startswith(prefix)}

def get_obsdata():
    directory_path = "D:/torchWorkspace/co2_ecbm/surrogate_model/data/dataSet"

    # 加载数据集
    data_tensors = load_datasets_from_h5(directory=directory_path)
    target_tensor = torch.tensor(data_tensors['data_con_co2'], dtype=torch.float32)
    
    # 添加一个维度，使形状变为 (4970, 12, 1, 32, 32)
    target_tensor = target_tensor.unsqueeze(2)

    # 选择特定的观测点，例如 (i, j)
    i, j = 1, 1  # 可以根据需要选择不同的点

    # 调用函数生成观测数据
    obs_data = create_synthetic_obs(target_tensor, i, j)

    # 将观测数据转换为 PyTorch 张量
    obs_data = torch.tensor(obs_data, dtype=torch.float32)

    # 展平张量
    obs_data = obs_data.flatten()

    return obs_data