import sys
import os
from torchinfo import summary

def setup_project_root():
    # 将项目根目录动态添加到 sys.path，确保模块可以被正确导入
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def main():
    # 动态设置项目路径
    setup_project_root()

    import torch
    from omegaconf import OmegaConf
    from models.SurrogateModel import SurrogateModel
    from utils.dataUtils import load_per_por_from_h5

    # 加载配置文件
    config_path = os.path.join("config", "SurrogateModel.yaml")
    task_name = 'task2'
    cfg = OmegaConf.load(config_path)
    cfg = cfg['tasks'][task_name]

    # 创建模型实例
    model = SurrogateModel(config=cfg)

    # 打印确认模型实例化成功
    print("Model has been instantiated successfully!")

    # 定义输入张量形状
    batch_size = 32  # 设置批量大小
    n_input_channels = 2  # 输入通道数（渗透率和孔隙率）
    time_history = cfg.model_params.time_history  # 时间步长
    height, width = 64, 64  # 假设输入为 64x64 的网格

    input_tensor_shape = (batch_size, time_history, n_input_channels, height, width)

    # 使用 torchinfo 打印模型结构
    print("Inspecting Model Structure:")
    summary(
        model,
        input_size=input_tensor_shape,  # 输入张量的形状
        col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"],
        depth=3,  # 控制输出的深度
    )

if __name__ == "__main__":
    main()
