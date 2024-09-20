import os
import pickle
import torch
import h5py

def save_pca_model(pca_model, model_path: str):
    """
    保存PCA模型。

    Args:
        pca_model (PCAModel): 训练好的PCA模型。
        model_path (str): 模型保存路径。
    """
    try:
        # 获取目录路径
        dir_path = os.path.dirname(model_path)
        
        # 如果目录不存在则创建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # 检查目录是否具有写权限
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"目录 {dir_path} 没有写入权限。")
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(pca_model, f)
        print(f"PCA模型已成功保存到 {model_path}")
    
    except PermissionError as e:
        print(f"权限错误：{e}")
    except FileNotFoundError as e:
        print(f"文件未找到错误：{e}")
    except Exception as e:
        print(f"发生错误：{e}")

def save_images_to_hdf5(images: torch.Tensor, reconstructed_images: torch.Tensor, file_path: str):
    """
    将原始图像和重构图像保存到HDF5文件中。

    Args:
        images (torch.Tensor): 原始图像数据。
        reconstructed_images (torch.Tensor): 重构后的图像数据。
        file_path (str): HDF5文件保存路径。
    """
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset('original', data=images)
        h5f.create_dataset('reconstructed', data=reconstructed_images)
    print(f"HDF5 file saved at {file_path}")