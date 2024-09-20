import sys
import os

def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)

def main():
    # 动态设置项目根目录
    setup_project_root()

    import torch
    from torch.utils.data import TensorDataset
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from config.FourierModelConfig import load_config
    from models.AHMModel import AHMModel
    from data.AHMDataModule import AHMDataModule
    from utils.dataUtils import load_datasets_from_h5

    # 设置为中等精度，以提高性能
    torch.set_float32_matmul_precision('medium')

    # 加载配置文件
    config_path = os.path.join("config", "FourierUNet.yaml")
    task_name = 'task1'
    cfg = load_config(config_path, task_name)

    # 读取配置文件中的数据路径和字段
    directory_path = cfg.task_params.data_set_path
    input_field_per = cfg.task_params.input_field1
    input_field_por = cfg.task_params.input_field2
    output_field_co2 = cfg.task_params.target_field

    # 加载数据集
    data_tensor = load_datasets_from_h5(directory=directory_path)

    # 从数据集中取得张量
    inputs_tensor_per = torch.tensor(data_tensor[input_field_per], dtype=torch.float32)
    inputs_tensor_por = torch.tensor(data_tensor[input_field_por], dtype=torch.float32)
    target_tensor = torch.tensor(data_tensor[output_field_co2], dtype=torch.float32)

    # 预处理数据
    conversion_factor = 1.01325e15
    inputs_tensor_per = inputs_tensor_per * conversion_factor  # 将渗透率张量转换为 mD
    inputs_tensor_per = inputs_tensor_per.unsqueeze(1)
    inputs_tensor_por = inputs_tensor_por.unsqueeze(1)

    inputs_tensor = torch.cat((inputs_tensor_per, inputs_tensor_por), dim=2)
    target_tensor = target_tensor.unsqueeze(2)

    # 打印数据的形状
    print(inputs_tensor.shape)  # shape: (4970, 1, 2, 32, 32)
    print(target_tensor.shape)  # shape: (4970, 12, 1, 32, 32)

    # 封装数据
    dataset = TensorDataset(inputs_tensor, target_tensor)

    # 创建数据模块
    data_module = AHMDataModule(dataset, batch_size=cfg.training_params.batch_size)

    # 创建模型实例，传递配置字典
    train_model = AHMModel(config=cfg.to_dict())

    # 设置日志记录器
    logger = TensorBoardLogger(
        save_dir=cfg.task_params.logger_path,
        name=cfg.task_params.logger_name
    )

    # 设置模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 验证集损失
        dirpath=logger.log_dir,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'  # 最小化验证损失
    )

    # 创建训练器并训练
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.training_params.max_epochs
    )
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
