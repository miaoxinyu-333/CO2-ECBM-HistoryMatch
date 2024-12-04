import torch
import os
import sys
# 获取项目根目录
def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)


def main():
    # Setup
    setup_project_root()
    import torch
    from models.ForwardModelCreator import ForwardModelCreator
    import torch
    from torch.utils.data import TensorDataset
    from omegaconf import OmegaConf
    from data.AHMDataModule import AHMDataModule
    from utils.dataUtils import load_co2_h5_file
    from utils.dataUtils import load_per_por_from_h5
    from evaluation.eval_metric import ssim_index
    from evaluation.eval_metric import r2_score_pytorch
    from evaluation.eval_metric import normalized_root_mean_squared_error
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置为中等精度，以提高性能
    torch.set_float32_matmul_precision('medium')

    # 加载配置文件
    config_path = os.path.join("config", "SurrogateModel.yaml")
    task_name = 'task1'
    cfg = OmegaConf.load(config_path)

    cfg = cfg['tasks'][task_name]

    per_por_file_path = "dataSet/surrogate/dataset_per_por_4983.h5"
    co2_file_path = "dataSet/surrogate/dataset_co2_4983.h5"

    per_tensor, por_tensor = load_per_por_from_h5(per_por_file_path)
    per_tensor = per_tensor.unsqueeze(1)
    por_tensor = por_tensor.unsqueeze(1)
    per_tensor = torch.log(per_tensor + 1e-8) 
    per_min, per_max = per_tensor.min(), per_tensor.max()
    per_tensor = (per_tensor - per_min) / (per_max - per_min)  # 归一化到 [0, 1]
    por_min, por_max = por_tensor.min(), por_tensor.max()
    por_tensor = (por_tensor - por_min) / (por_max - por_min)  # 归一化到 [0, 1]
    inputs_tensor = torch.cat((per_tensor, por_tensor), dim=1)
    co2_tensor = load_co2_h5_file(co2_file_path)
    target_tensor = co2_tensor.unsqueeze(2)

    print(inputs_tensor.shape)

    dataset = TensorDataset(inputs_tensor, target_tensor)

    # 创建数据模块
    data_module = AHMDataModule(dataset, batch_size=cfg.training_params.batch_size)
    # 假设你的 autoencoder_checkpoint 路径正确
    autoencoder_checkpoint_path = "logs/AutoencoderKL_PTH/best_autoencoder_model.pth"  # 替换为实际的路径

    # 创建 ForwardModelCreator 实例，指定设备（GPU 或 CPU）
    creator = ForwardModelCreator(device='cuda')  # 如果没有GPU，使用 'cpu'

    # 创建前向模型
    forward_model = creator.create_forward_model(autoencoder_checkpoint_path)

    # 打印模型结构，验证是否已正确加载
    print("Forward model created successfully!")

    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    # 初始化变量来计算平均 NRMSE、R² 和 SSIM
    total_nrmse = 0.0
    total_r2 = 0.0
    total_ssim = 0.0
    total_mse = 0.0  # 初始化 MSE 总和
    num_batches = 0

    # 前向推理
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 模型的前向推理
            predictions = forward_model(inputs)
            
            # 计算 NRMSE
            batch_nrmse = normalized_root_mean_squared_error(targets, predictions)
            total_nrmse += batch_nrmse.item()
            
            # 计算 R²
            batch_r2 = r2_score_pytorch(targets, predictions)
            total_r2 += batch_r2
            
            # 计算 SSIM
            batch_ssim = ssim_index(targets, predictions)
            total_ssim += batch_ssim

            batch_mse = torch.mean((targets - predictions) ** 2).item()
            total_mse += batch_mse

            num_batches += 1

    # 计算各指标的平均值
    mean_nrmse = total_nrmse / num_batches
    mean_r2 = total_r2 / num_batches
    mean_ssim = total_ssim / num_batches
    mean_mse = total_mse / num_batches  # 计算平均 MSE


    # 输出结果
    print(f"Mean NRMSE over the test set: {mean_nrmse:.4f}")
    print(f"Mean R² over the test set: {mean_r2:.4f}")
    print(f"Mean SSIM over the test set: {mean_ssim:.4f}")
    print(f"Mean MSE over the test set: {mean_mse:.4f}")

if __name__ == "__main__":
    main()
