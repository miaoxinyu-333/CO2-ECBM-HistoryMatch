import os
import sys
from torchinfo import summary

def setup_project_root():
    # 将项目根目录添加到 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def main():
    # 动态设置项目根目录
    setup_project_root()

    import torch
    from omegaconf import OmegaConf
    from models.SurrogateModel import SurrogateModel

    # 加载配置文件
    config_path = os.path.join("config", "SurrogateModel.yaml")
    task_name = 'task1'
    cfg = OmegaConf.load(config_path)
    cfg = cfg['tasks'][task_name]

    # 创建模型实例
    model = SurrogateModel(config=cfg)

    # 检查模型是否被正确实例化
    print("Model has been instantiated successfully!")

    # 定义输入形状
    batch_size = 32  # 根据实际训练时的批量大小
    time_steps = cfg.model_params.time_history  # 时间步数，从配置中获取
    n_input_channels = cfg.model_params.n_input_scalar_components  # 输入通道数
    height, width = 64, 64  # 假设输入为 64x64 的网格

    # 使用 torchinfo 检查模型详细结构
    print("Model Summary:")
    summary(
        model,
        input_size=(batch_size, time_steps, n_input_channels, height, width),  # 输入张量的形状
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],  # 输出列
        depth=3  # 控制输出深度
    )

if __name__ == "__main__":
    main()
