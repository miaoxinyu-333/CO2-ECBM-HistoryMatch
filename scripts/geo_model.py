import gstools as gs
import numpy as np
import pandas as pd
import os

# 创建文件夹来保存CSV文件
output_folder_perm = '../dataSet/permeability_datasets'
output_folder_poro = '../dataSet/porosity_datasets'
os.makedirs(output_folder_perm, exist_ok=True)
os.makedirs(output_folder_poro, exist_ok=True)

# 创建坐标范围从0到199，网格大小为1x1，总尺寸200x200
x = y = range(200)

# 定义高斯模型
model = gs.Gaussian(dim=2, var=1, len_scale=10)

# 遍历seed值从1到5000
for seed in range(1, 5001):
    # 初始化随机场生成器
    srf = gs.SRF(model, seed=seed)

    # 生成结构化随机场
    field = srf.structured((x, y))

    # 渗透率的处理，使用指数转换并映射到0.5到5mD
    positive_field_perm = np.exp(field - np.max(field))  # 应用指数转换确保正值
    min_perm = 0.5
    max_perm = 5
    scaled_field_perm = min_perm + (max_perm - min_perm) * (positive_field_perm / np.max(positive_field_perm))

    # 孔隙度的处理，线性映射到2%到6%
    min_poro = 0.02
    max_poro = 0.06
    scaled_field_poro = min_poro + (max_poro - min_poro) * (field - np.min(field)) / (np.max(field) - np.min(field))

    # 创建数据集，包含X, Y坐标和渗透率
    coordinates = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    permeability = scaled_field_perm.flatten()
    porosity = scaled_field_poro.flatten()
    data_perm = np.column_stack((coordinates, permeability))
    data_poro = np.column_stack((coordinates, porosity))

    # 转换为pandas DataFrame，并保存为CSV文件
    df_perm = pd.DataFrame(data_perm, columns=['X', 'Y', 'Permeability'])
    df_poro = pd.DataFrame(data_poro, columns=['X', 'Y', 'Porosity'])

    csv_file_path_perm = os.path.join(output_folder_perm, f'{seed}.csv')
    csv_file_path_poro = os.path.join(output_folder_poro, f'{seed}.csv')

    df_perm.to_csv(csv_file_path_perm, index=False)
    df_poro.to_csv(csv_file_path_poro, index=False)

    # 打印进度信息
    if seed % 100 == 0:
        print(f'CSV files saved to {csv_file_path_perm} and {csv_file_path_poro}, progress: {seed}/5000')

# 结束生成
print("All permeability and porosity data sets generated successfully.")
