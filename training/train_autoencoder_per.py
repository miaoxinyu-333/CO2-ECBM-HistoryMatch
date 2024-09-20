import os
import sys

def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)

def main():
    # 动态添加项目根目录到 sys.path
    setup_project_root()

    # 导入需要的模块
    import torch
    from torch.utils.data import TensorDataset
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from models.AHMModel import AHMModel
    from data.AHMDataModule import AHMDataModule
    from utils.dataUtils import load_reconstuction_h5
    from config.FourierModelConfig import load_config

    # 设置为中等精度，以提高性能
    torch.set_float32_matmul_precision('medium')

    # 加载配置文件
    config_path = os.path.join("config", "FourierUNet.yaml")
    task_name = 'task2'
    cfg = load_config(config_path, task_name)

    # 加载和预处理数据
    data_set_path = cfg.task_params.data_set_path
    inputs_tensor, target_tensor = load_reconstuction_h5(data_set_path)
    
    print(inputs_tensor.shape)
    print(target_tensor.shape)

    # 封装数据
    dataset = TensorDataset(inputs_tensor, target_tensor)

    # 创建数据模块
    batch_size = cfg.training_params.batch_size
    data_module = AHMDataModule(dataset, batch_size=batch_size)

    # 设置日志记录器
    logger_path = cfg.task_params.logger_path
    logger_name = cfg.task_params.logger_name
    logger = TensorBoardLogger(save_dir=logger_path, name=logger_name)

    # 设置模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=logger.log_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # 创建训练器
    max_epochs = cfg.training_params.max_epochs
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs
    )

    # 创建模型实例，传递配置字典
    train_model = AHMModel(config=cfg.to_dict())  # 只在这里传递配置字典

    # 训练模型
    trainer.fit(train_model, datamodule=data_module)

    # 从检查点文件中提取模型状态字典
    best_model_path = checkpoint_callback.best_model_path
    checkpoint = torch.load(best_model_path)
    train_model.load_state_dict(checkpoint['state_dict'])

    # 保存最佳模型参数
    torch.save(train_model.state_dict(), cfg.task_params.save_model_path)

    # 测试模型
    trainer.test(train_model, datamodule=data_module, ckpt_path=best_model_path)

if __name__ == "__main__":
    main()
