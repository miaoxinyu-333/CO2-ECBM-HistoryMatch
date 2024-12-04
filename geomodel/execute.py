import gstools as gs
import numpy as np
import pandas as pd
import os

# 创建输出文件夹
output_folder_perm = 'permeability_datasets'
output_folder_poro = 'porosity_datasets'
os.makedirs(output_folder_perm, exist_ok=True)
os.makedirs(output_folder_poro, exist_ok=True)

# 定义坐标范围
x = y = range(64)

# 定义高斯模型
model = gs.Gaussian(dim=2, var=0.5, len_scale=8)

# 开始生成样本
for seed in range(1, 5001):
    # 初始化随机场生成器
    srf = gs.SRF(model, seed=seed)
    
    # 生成结构化随机场
    field = srf.structured((x, y))
    
    # 渗透率的处理，使用指数转换并映射到0.5到1mD
    positive_field_perm = np.exp(field - np.max(field))  # 应用指数转换确保正值
    min_perm, max_perm = 0.5, 1
    scaled_field_perm = min_perm + (max_perm - min_perm) * (positive_field_perm / np.max(positive_field_perm))
    
    # 孔隙度的处理，线性映射到5%到10%
    min_poro, max_poro = 0.05, 0.1
    scaled_field_poro = min_poro + (max_poro - min_poro) * (field - np.min(field)) / (np.max(field) - np.min(field))
    
    # 创建坐标数组
    coordinates = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    permeability = scaled_field_perm.flatten()
    porosity = scaled_field_poro.flatten()
    
    # 组合数据集
    data_perm = np.column_stack((coordinates, permeability))
    data_poro = np.column_stack((coordinates, porosity))
    
    # 转换为 pandas DataFrame
    df_perm = pd.DataFrame(data_perm, columns=['X', 'Y', 'Permeability'])
    df_poro = pd.DataFrame(data_poro, columns=['X', 'Y', 'Porosity'])
    
    # 保存为 CSV 文件
    csv_file_path_perm = os.path.join(output_folder_perm, f'{seed}.csv')
    csv_file_path_poro = os.path.join(output_folder_poro, f'{seed}.csv')
    
    df_perm.to_csv(csv_file_path_perm, index=False)
    df_poro.to_csv(csv_file_path_poro, index=False)
    
    # 打印进度信息
    if seed % 100 == 0:
        print(f'CSV files saved for sample {seed}/5000')

print("All permeability and porosity datasets generated successfully.")
