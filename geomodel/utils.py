import os
import numpy as np
import pandas as pd
import h5py
import shutil


def extract_tensors_from_txt(file_path):
    # (保持原有逻辑，不做修改)
    tensors = []
    reading_data = False  # 用于标记是否开始读取数据
    current_tensor = None
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith('%'):
                if "Data" in line:
                    reading_data = True  # 开始读取数据
                elif reading_data and current_tensor is not None:
                    tensors.append(np.array(current_tensor, dtype=float))
                    current_tensor = None
                continue
            if not reading_data:
                continue
            if current_tensor is None:
                current_tensor = []
            line = line.strip().split()
            current_tensor.append([float(num) for num in line])
        if current_tensor is not None:
            tensors.append(np.array(current_tensor, dtype=float))
    return np.array(tensors)


def save_tensor_to_h5(tensor, h5_filename):
    with h5py.File(h5_filename, 'w') as hdf:
        hdf.create_dataset('data', data=tensor)
        print(f"Data successfully saved to {h5_filename}.")

def process_and_save_data(dataset_path, folder_names):
    for folder in folder_names:
        folder_path = os.path.join(dataset_path, folder)
        all_tensors = []  # 用来存储所有文件的张量
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                tensor = extract_tensors_from_txt(file_path)
                all_tensors.append(tensor)  # 添加单个文件的张量
        
        # 将所有张量堆叠成一个大张量
        big_tensor = np.stack(all_tensors, axis=0)
        # 指定 HDF5 文件名
        h5_filename = os.path.join(dataset_path, f"{folder}.h5")
        save_tensor_to_h5(big_tensor, h5_filename)


def save_per_por_to_h5(output_folder_perm, output_folder_poro, h5_file_path, sample_shape):
    # 创建HDF5文件
    with h5py.File(h5_file_path, 'w') as h5f:
        # 创建两个数据集：渗透率和孔隙度
        perm_dset = h5f.create_dataset(
            'permeability', (len(os.listdir(output_folder_perm)), *sample_shape), dtype='float32'
        )
        poro_dset = h5f.create_dataset(
            'porosity', (len(os.listdir(output_folder_poro)), *sample_shape), dtype='float32'
        )

        # 获取文件夹中所有的csv文件
        perm_files = [f for f in os.listdir(output_folder_perm) if f.endswith('.csv')]
        poro_files = [f for f in os.listdir(output_folder_poro) if f.endswith('.csv')]

        # 确保渗透率和孔隙度文件数量一致
        if len(perm_files) != len(poro_files):
            print(f"Warning: The number of permeability files ({len(perm_files)}) does not match the number of porosity files ({len(poro_files)}).")
            return
        
        # 遍历文件夹中的所有csv文件
        for idx, perm_file in enumerate(perm_files):
            perm_file_path = os.path.join(output_folder_perm, perm_file)
            poro_file_path = os.path.join(output_folder_poro, poro_files[idx])

            # 读取渗透率数据
            perm_df = pd.read_csv(perm_file_path)
            perm_data = perm_df['Permeability'].values.reshape(sample_shape)

            # 读取孔隙度数据
            poro_df = pd.read_csv(poro_file_path)
            poro_data = poro_df['Porosity'].values.reshape(sample_shape)

            # 将数据写入HDF5文件
            perm_dset[idx, :, :] = perm_data
            poro_dset[idx, :, :] = poro_data

            # 进度提示
            if (idx + 1) % 100 == 0:
                print(f'Saved {idx + 1}/{len(perm_files)} samples to HDF5.')

        print(f'HDF5 file saved at {h5_file_path}')

def save_co2_to_h5(dataset_path, folder_name, output_h5_filename):
    folder_path = os.path.join(dataset_path, folder_name)
    all_tensors = []  # 用来存储所有文件的张量

    # 遍历文件夹中的所有 .txt 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # 调用原来的 extract_tensors_from_txt 函数
            tensor = extract_tensors_from_txt(file_path)
            all_tensors.append(tensor)  # 将读取的张量添加到列表中
    
    # 将所有张量堆叠成一个大张量
    big_tensor = np.stack(all_tensors, axis=0)  # 使用 vstack 将所有的张量堆叠成一个大数组
    
    # 保存堆叠后的大张量到 HDF5 文件
    with h5py.File(output_h5_filename, 'w') as h5f:
        h5f.create_dataset('data', data=big_tensor)
    
    print(f"Saved stacked tensor to {output_h5_filename}")