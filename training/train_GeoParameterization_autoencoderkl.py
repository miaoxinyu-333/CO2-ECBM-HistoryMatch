import os
import sys
import torch

def setup_project_root():
    # 将项目根目录动态添加到 sys.path，确保模块可以被正确导入
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def main():
    # 动态设置项目路径
    setup_project_root()

    from torch.utils.data import TensorDataset
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from models.AutoencoderKLWrapper import AutoencoderKLWrapper
    from data.AHMDataModule import AHMDataModule
    from utils.dataUtils import load_per_por_from_h5

    # 设置 PyTorch 性能优化
    torch.set_float32_matmul_precision('medium')

    # 明确参数配置
    params = {
        "data_set_path": "dataSet/surrogate/dataset_per_por_4983.h5",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "kl_weight": 0.0000001,                        # KL 散度的权重
        "weight_decay": 0,                    # 权重衰减
        "max_epochs": 100,
        "logger_path": "D:\\torchWorkspace\\CO2-ECBM-HistoryMatch\\logs\\tb_logs",
        "logger_name": "AutoencoderKL",
        "save_model_path": "D:\\torchWorkspace\\CO2-ECBM-HistoryMatch\\logs\\AutoencoderKL_PTH\\best_autoencoder_model.pth"
    }

    per_tensor, por_tensor = load_per_por_from_h5(params["data_set_path"])

    per_tensor = per_tensor.unsqueeze(1)
    por_tensor = por_tensor.unsqueeze(1)
    per_tensor = torch.log(per_tensor + 1e-8)  # 防止 log(0) 的情况
    per_min, per_max = per_tensor.min(), per_tensor.max()
    per_tensor = (per_tensor - per_min) / (per_max - per_min)  # 归一化到 [0, 1]
    por_min, por_max = por_tensor.min(), por_tensor.max()
    por_tensor = (por_tensor - por_min) / (por_max - por_min)  # 归一化到 [0, 1]
    inputs_tensor = torch.cat((per_tensor, por_tensor), dim=1)
    target_tensor = inputs_tensor

    # 创建 PyTorch 数据集
    dataset = TensorDataset(inputs_tensor, target_tensor)

    # 初始化数据模块
    data_module = AHMDataModule(dataset, batch_size=params["batch_size"])

    # 初始化 TensorBoard 日志记录器
    logger = TensorBoardLogger(save_dir=params["logger_path"], name=params["logger_name"])

    # 设置模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=logger.log_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # 创建 PyTorch Lightning Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=params["max_epochs"]
    )

    # 初始化 AutoencoderKL 模型
    autoencoder_model = AutoencoderKLWrapper(
        lr=params["learning_rate"],
        kl_weight=params["kl_weight"],
        weight_decay=params["weight_decay"]
    )

    # 开始训练
    trainer.fit(autoencoder_model, datamodule=data_module)

    # 提取最佳模型权重
    best_model_path = checkpoint_callback.best_model_path
    checkpoint = torch.load(best_model_path)
    autoencoder_model.load_state_dict(checkpoint['state_dict'])

    # 保存最佳模型
    torch.save(autoencoder_model.state_dict(), params["save_model_path"])

    # 测试模型
    trainer.test(autoencoder_model, datamodule=data_module, ckpt_path=best_model_path)

if __name__ == "__main__":
    main()
