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


# 加载permeability和porosity的函数，返回PyTorch张量
def load_per_por_from_h5(h5_file_path):
    """
    从指定的 HDF5 文件中加载渗透率 (permeability) 和孔隙度 (porosity) 数据集。
    返回两个PyTorch张量：一个包含渗透率数据，一个包含孔隙度数据。
    """
    with h5py.File(h5_file_path, 'r') as h5f:
        # 获取渗透率和孔隙度数据集，并转换为PyTorch张量
        perm_data = torch.tensor(h5f['permeability'][:], dtype=torch.float32)
        poro_data = torch.tensor(h5f['porosity'][:], dtype=torch.float32)
    
    return perm_data, poro_data

# 加载CO2数据的函数
def load_co2_h5_file(h5_file_path):
    """
    从指定的 HDF5 文件中加载 CO2 数据集，并返回 PyTorch 张量。
    """
    with h5py.File(h5_file_path, 'r') as h5f:
        # 假设CO2数据存储在 'data' 数据集
        co2_data = torch.tensor(h5f['data'][:], dtype=torch.float32)
    
    return co2_data


def load_autoencoderkl_h5(file_path):
    """
    从 HDF5 文件中加载输入和目标张量。

    Args:
        file_path (str): HDF5 文件的路径。
    
    Returns:
        tuple: 包含两个 PyTorch 张量 (inputs_tensor, target_tensor)。
    """
    try:
        with h5py.File(file_path, 'r') as h5f:
            # 提取输入和目标数据集
            inputs = h5f['input'][:]
            targets = h5f['target'][:]
        
        # 转换为 PyTorch 张量
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        return inputs_tensor, target_tensor
    except Exception as e:
        print(f"Error while loading HDF5 file: {e}")
        raise

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
    return model_output[:, 0, 3, 3]

def get_obsdata(h5_file_path, i=3, j=3):
    """
    根据指定 HDF5 文件路径加载 CO2 数据集，生成观测数据。
    
    Args:
        h5_file_path (str): HDF5 文件路径。
        i (int): 观测点的第一个索引。
        j (int): 观测点的第二个索引。
    
    Returns:
        torch.Tensor: 展平后的观测数据。
    """
    # 加载数据集
    target_tensor = load_co2_h5_file(h5_file_path)
    
    target_tensor = target_tensor.unsqueeze(2)

    # 调用函数生成观测数据
    obs_data = create_synthetic_obs(target_tensor, i, j)

    # 将观测数据转换为 PyTorch 张量
    obs_data = torch.tensor(obs_data, dtype=torch.float32)

    # 展平张量
    obs_data = obs_data.flatten()

    return obs_data


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

def inspect_pth_file(pth_file_path):
    # 加载 PTH 文件
    checkpoint = torch.load(pth_file_path, map_location='cpu')  # 将其加载到 CPU 上，避免 GPU 相关的问题

    # 打印 PTH 文件的整体结构
    print(f"Contents of the PTH file at {pth_file_path}:")
    
    # 如果 PTH 文件是包含 state_dict 的形式
    if 'state_dict' in checkpoint:
        print("\nKeys in state_dict:")
        for key in checkpoint['state_dict'].keys():
            print(key)
        
        # 如果需要，还可以直接打印出某些层的权重
        print("\nFirst few parameters:")
        for name, param in checkpoint['state_dict'].items():
            print(f"{name}: {param.shape}")
            break  # 只显示第一个参数的形状，您可以删除这行显示所有参数
        
    else:
        print("\nThe PTH file does not contain a state_dict, checking contents:")
        for key, value in checkpoint.items():
            print(f"{key}: {value}")