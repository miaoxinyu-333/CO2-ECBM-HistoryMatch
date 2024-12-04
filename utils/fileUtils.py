import os
import pickle
import torch
import h5py

def load_pca_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

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


def extract_tensors_from_txt(file_path):
    tensors = []
    reading_data = False  # 用于标记是否开始读取数据
    current_tensor = None
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # 如果行以 '%' 开头，则表示注释
            if line.startswith('%'):
                if "Data" in line:
                    reading_data = True  # 开始读取数据
                # 如果已经在读取数据，则将当前张量添加到张量列表中
                elif reading_data and current_tensor is not None:
                    tensors.append(current_tensor)
                    current_tensor = None
                continue
            # 如果尚未开始读取数据，则继续跳过
            if not reading_data:
                continue
            # 如果当前张量为空，则创建一个新的空张量
            if current_tensor is None:
                current_tensor = []
            # 去除行末的换行符并将数字以空格分割
            line = line.strip().split()
            # 将每行的数字转换为浮点数并添加到当前张量中
            current_tensor.append([float(num) for num in line])
        # 将最后一个张量添加到张量列表中
        if current_tensor is not None:
            tensors.append(current_tensor)
    return tensors

def save_tensor_to_h5(tensor, h5_filename):
    with h5py.File(h5_filename, 'w') as hdf:
        hdf.create_dataset('data', data=tensor)
        print(f"Data successfully saved to {h5_filename}.")

def extract_tensors_from_folder(folder_path, ends_name=".txt"):
    concatenated_tensor = []
    for filename in os.listdir(folder_path):
        if filename.endswith(ends_name):
            file_path = os.path.join(folder_path, filename)
            tensors_data = extract_tensors_from_txt(file_path)
            concatenated_tensor.append(tensors_data)
    return concatenated_tensor