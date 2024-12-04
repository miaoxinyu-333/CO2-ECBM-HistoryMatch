import mph
import os

# 创建 COMSOL 客户端连接
client = mph.Client()

# 加载 COMSOL 模型
model_path = 'D:\\comsol\\model\\CO2_ECBM_2.mph'
model = client.load(model_path)

# 文件夹路径和数据节点定义
data_set_path = "D:\\torchWorkspace\\CO2-ECBM-HistoryMatch\\logs\\sensitivity_analysis"
co2_concentration_node = "data_con_co2"  # 关注 CO2 浓度
data_path_count = "/{}_{}.txt"

# 基准参数设置
baseline_params = {
    'poro_pore_0': '0.1',  # 基准孔隙率
    'K0': '1.0 [mD]',  # 基准渗透率
    'P1_0': '2.5 [MPa]',  # 基准 CH4 压力
    'T_0': '310.15 [K]'  # 基准温度
}

# 敏感性分析参数（每个参数的不同值）
import numpy as np

# 基准参数设置（保持不变）
baseline_params = {
    'poro_pore_0': '0.1',  # 基准孔隙率
    'K0': '1.0 [mD]',      # 基准渗透率
    'P1_0': '2.5 [MPa]',   # 基准 CH4 压力
    'T_0': '310.15 [K]'    # 基准温度
}

# 敏感性分析参数（动态生成范围内的等间隔点）
sensitivity_params = {
    'poro_pore_0': [f"{v:.2f}" for v in np.linspace(0.05, 0.15, 5)],  # 生成等间隔点
    'K0': [f"{v:.2f} [mD]" for v in np.linspace(0.5, 1.5, 5)],
    'P1_0': [f"{v:.2f} [MPa]" for v in np.linspace(2.0, 3.0, 5)],
    'T_0': [f"{v:.2f} [K]" for v in np.linspace(305.5, 315.5, 5)]
}

# 打印生成的参数
for param, values in sensitivity_params.items():
    print(f"{param}: {values}")


# 创建敏感性分析结果文件夹
if not os.path.exists(data_set_path):
    os.makedirs(data_set_path)

# 记录输出文件编号
count = 1

# 运行基准模型

# 将所有参数设置为基准状态
print("Running baseline model...")
for param_name, base_value in baseline_params.items():
    model.parameter(param_name, base_value)

try:
    # 模型运行
    model.solve(study="bigData")
    # 导出基准 CO2 浓度数据
    baseline_output_path = os.path.join(data_set_path, "baseline.txt")
    model.export(node=co2_concentration_node, file=baseline_output_path)
    print(f"Baseline simulation completed and result saved at {baseline_output_path}")
except Exception as e:
    print(f"Exception occurred during baseline simulation: {str(e)}")


# 进行敏感性分析，逐一调整每个参数
for param_name, values in sensitivity_params.items():
    for value in values:
        try:
            # 每次开始前将所有参数重置为基准值
            for base_param, base_value in baseline_params.items():
                model.parameter(base_param, base_value)

            # 修改当前参数以进行敏感性分析
            model.parameter(param_name, value)

            # 模型运行
            model.solve(study="bigData")

            # 导出 CO2 浓度数据
            output_path = os.path.join(data_set_path, f'{param_name}_{value}.txt')
            print(output_path)
            model.export(node=co2_concentration_node, file=output_path)

            print(f"Simulation {count} for {param_name}={value} completed and result saved.")
            count += 1
        except Exception as e:
            print(f"Exception occurred for {param_name}={value}: {str(e)}")

print("Sensitivity analysis completed successfully.")

model.save()

client.disconnect()
