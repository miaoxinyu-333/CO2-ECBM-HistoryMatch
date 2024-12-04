import os
import sys
from torchinfo import summary
import torch
import time

def setup_project_root():
    # 将项目根目录添加到 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def measure_inference_time(model, input_tensor, num_runs=100):
    """
    使用 torch.cuda.Event 测量模型的推理时间。
    Args:
        model (torch.nn.Module): 待测模型
        input_tensor (torch.Tensor): 输入张量
        num_runs (int): 测试运行次数
    Returns:
        float: 平均推理时间（毫秒）
    """
    # 确保模型在评估模式下
    model.eval()

    # 初始化 CUDA 时间事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 预热 GPU（避免首次运行的延迟影响结果）
    with torch.no_grad():
        _ = model(input_tensor)

    # 测量多次推理时间
    total_time = 0.0
    with torch.no_grad():
        for _ in range(num_runs):
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()  # 确保计算完成
            total_time += start_event.elapsed_time(end_event)  # 获取时间间隔

    # 计算平均时间
    avg_time = total_time / num_runs
    return avg_time

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
    model = SurrogateModel(config=cfg).cuda()  # 模型移至 GPU
    print("Model has been instantiated successfully!")

    # 定义输入形状
    batch_size = 32  # 根据实际训练时的批量大小
    time_steps = cfg.model_params.time_history  # 时间步数，从配置中获取
    n_input_channels = cfg.model_params.n_input_scalar_components  # 输入通道数
    height, width = 64, 64  # 假设输入为 64x64 的网格

    # 创建输入张量并移动到 GPU
    input_tensor = torch.randn(batch_size, time_steps, n_input_channels, height, width).cuda()

    # 测量批次推理时间
    avg_batch_time = measure_inference_time(model, input_tensor, num_runs=100)
    print(f"Average Batch Inference Time: {avg_batch_time:.3f} ms")

    # 计算单个样本的推理时间
    avg_sample_time = avg_batch_time / batch_size
    print(f"Average Inference Time per Sample: {avg_sample_time:.3f} ms")

if __name__ == "__main__":
    main()
