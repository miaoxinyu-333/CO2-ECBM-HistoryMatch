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

    latent_variables = []  # 用于存储所有测试样本的潜在变量

    # 遍历测试数据集
    forward_model.eval()  # 确保模型处于评估模式
    with torch.no_grad():  # 禁用梯度计算，提高效率
        for batch in test_loader:
            inputs, _ = batch  # inputs 是形状为 (batch_size, 2, 64, 64) 的张量
            inputs = inputs.to(device)
            
            # 前向传播获取潜在变量
            latents = forward_model(inputs)  # 确保 forward_model 返回的是潜在变量
            latent_variables.append(latents.cpu())  # 将潜在变量移动到 CPU，并存储

    # 将潜在变量合并为一个张量
    latent_variables = torch.cat(latent_variables, dim=0)  # (num_samples, latent_dim)
    print("Collected latent variables shape:", latent_variables.shape)

    import matplotlib.pyplot as plt

    # 将潜在变量展平为二维数组 (num_samples, latent_dim)
    latent_variables_flat = latent_variables.view(latent_variables.size(0), -1)

    # 遍历每个维度
    for i in range(latent_variables_flat.size(1)):  # 遍历每个潜在变量维度
        plt.hist(latent_variables_flat[:, i].numpy(), bins=50, alpha=0.6, label=f"Latent dim {i+1}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of latent dimension {i+1}")
        plt.show()


if __name__ == "__main__":
    main()
