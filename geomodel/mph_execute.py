import mph
import os
import shutil  # 用于文件复制

# 创建 COMSOL 客户端连接
client = mph.Client()

# 加载 COMSOL 模型
model_path = 'D:\\comsol\\model\\CO2_ECBM.mph'
model = client.load(model_path)

# 获取脚本运行的当前目录
current_dir = os.getcwd()
root_folder = os.path.join(current_dir, "surrogate_dataset")

# 创建子文件夹
permeability_folder = os.path.join(root_folder, 'permeability_datasets')
porosity_folder = os.path.join(root_folder, 'porosity_datasets')
data_con_co2_folder = os.path.join(root_folder, 'data_con_co2')

# 确保文件夹存在
os.makedirs(permeability_folder, exist_ok=True)
os.makedirs(porosity_folder, exist_ok=True)
os.makedirs(data_con_co2_folder, exist_ok=True)

iter_start = 1
iter_end = 5001

for i in range(iter_start, iter_end):
    # 构建两种数据文件的路径
    source_permeability_file_path = os.path.join('permeability_datasets', f'{i}.csv')
    source_porosity_file_path = os.path.join('porosity_datasets', f'{i}.csv')
    target_permeability_file_path = os.path.join(permeability_folder, f'{i}.csv')
    target_porosity_file_path = os.path.join(porosity_folder, f'{i}.csv')

    try:
        node = model/'functions'
        node_por = node/'por_int'
        node_per = node/'per_int'

        # 设置插值函数的文件路径
        node_por.property('filename', value=source_porosity_file_path)
        node_per.property('filename', value=source_permeability_file_path)

        # 模型运行
        model.solve(study="bigData")

        # 导出 data_con_co2 数据到目标文件夹
        data_con_co2_file_path = os.path.join(data_con_co2_folder, f'{i}.txt')
        model.export(node="data_con_co2", file=data_con_co2_file_path)

        # 复制 permeability 和 porosity CSV 文件到目标文件夹
        shutil.copy(source_permeability_file_path, target_permeability_file_path)
        shutil.copy(source_porosity_file_path, target_porosity_file_path)

        print(f"Simulation {i} completed and result saved.")

    except Exception as e:
        print(f"Exception occurred for simulation {i}: {str(e)}")

    # 打印进度
    if i % 100 == 0:
        print(f'Simulation {i} completed and results saved.')

print("All simulations completed successfully.")

model.save()
