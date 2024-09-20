import mph
import os

# 创建 COMSOL 客户端连接
client = mph.Client()

# 加载 COMSOL 模型
model_path = 'D:\\comsol\\model\\CO2_ECBM.mph'
model = client.load(model_path)

data_set_path = "../dataSet/pycomsol"
data_node1 = "data_con_ch4"
data_node2 = "data_con_co2"
data_node3 = "data_es"
data_node4 = "data_por"
data_node5 = "data_per"
data_node6 = "data_T"
data_path_count = "/{}.txt"

# 设定文件夹路径
permeability_folder = 'permeability_datasets'
porosity_folder = 'porosity_datasets'

count = 1
# 循环执行2000次仿真

iter_start = 1
iter_end = 5001

for i in range(iter_start, iter_end):
    # 构建两种数据文件的路径
    permeability_file_path = os.path.join(permeability_folder, f'{i}.csv')
    porosity_file_path = os.path.join(porosity_folder, f'{i}.csv')

    try:
        node = model/'functions'
        node_por = node/'por_int'
        node_per = node/'per_int'
        # 设置插值函数的文件路径
        node_por.property('filename', value=porosity_file_path)
        node_per.property('filename', value=permeability_file_path)

        # 模型运行
        model.solve(study="bigData")

        # 导出数据
        model.export(node=data_node1, file=data_set_path + data_node1 + data_path_count.format(count))
        model.export(node=data_node2, file=data_set_path + data_node2 + data_path_count.format(count))
        model.export(node=data_node3, file=data_set_path + data_node3 + data_path_count.format(count))
        model.export(node=data_node4, file=data_set_path + data_node4 + data_path_count.format(count))
        model.export(node=data_node5, file=data_set_path + data_node5 + data_path_count.format(count))
        model.export(node=data_node6, file=data_set_path + data_node6 + data_path_count.format(count))

        print("result {} is finish".format(count))
        count = count + 1
    except Exception as e:
        print(f"Exception occurred for simulation {i} CSV file: {str(e)}")

    # 打印进度
    if i % 100 == 0:
        print(f'Simulation {i} completed and results saved.')

print("All simulations completed successfully.")

model.save()