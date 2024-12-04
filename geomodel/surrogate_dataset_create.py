from utils import extract_tensors_from_txt
import os
import numpy as np
import h5py
import pandas as pd
from utils import save_per_por_to_h5
from utils import save_co2_to_h5


dataset_path = 'D:/torchWorkspace/co2_ecbm/pycomsol4/surrogate_dataset'
folder_name = 'data_con_co2'  # 替换成实际的文件夹名
output_h5_filename = 'dataset_co2_4983.h5'  # 输出的 HDF5 文件名

# 调用函数
save_co2_to_h5(dataset_path, folder_name, output_h5_filename)

# 定义文件夹路径
output_folder_perm = 'surrogate_dataset/permeability_datasets'
output_folder_poro = 'surrogate_dataset/porosity_datasets'

# 定义保存的HDF5文件路径
h5_file_path = 'dataset_per_por_4983.h5'
sample_shape = (64, 64) 

save_per_por_to_h5(output_folder_perm, output_folder_poro, h5_file_path, sample_shape)