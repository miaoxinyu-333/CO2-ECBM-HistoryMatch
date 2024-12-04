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
    from omegaconf import OmegaConf
    from models.SurrogateModel import SurrogateModel
    from data.AHMDataModule import AHMDataModule
    from utils.dataUtils import load_co2_h5_file
    from utils.dataUtils import load_per_por_from_h5
    from utils.dataUtils import load_reconstuction_h5

    # 设置为中等精度，以提高性能
    torch.set_float32_matmul_precision('medium')

    # 加载配置文件
    config_path = os.path.join("config", "SurrogateModel.yaml")
    task_name = 'task3'
    cfg = OmegaConf.load(config_path)

    cfg = cfg['tasks'][task_name]

    """
    # 文件路径
    per_file_path = "dataSet/reconstruction/reconstructions_dataset_per.h5"
    por_file_path = "dataSet/reconstruction/reconstructions_dataset_por.h5"

    per_tensor, _ = load_reconstuction_h5(per_file_path)
    por_tensor, _ = load_reconstuction_h5(por_file_path)

    inputs_tensor = torch.cat((per_tensor, por_tensor), dim=2)

    co2_file_path = "dataSet/raw/data_con_co2.h5"

    target_tensor = load_co2_h5_file(co2_file_path)
    target_tensor = target_tensor.unsqueeze(2)
    """

    per_por_file_path = "dataSet/surrogate/dataset_per_por_4983.h5"
    co2_file_path = "dataSet/surrogate/dataset_co2_4983.h5"

    per_tensor, por_tensor = load_per_por_from_h5(per_por_file_path)
    per_tensor = per_tensor.unsqueeze(1).unsqueeze(2)
    por_tensor = por_tensor.unsqueeze(1).unsqueeze(2)
    inputs_tensor = torch.cat((per_tensor, por_tensor), dim=2)
    co2_tensor = load_co2_h5_file(co2_file_path)
    target_tensor = co2_tensor.unsqueeze(2)

    print(inputs_tensor.shape)
    print(target_tensor.shape)

    dataset = TensorDataset(inputs_tensor, target_tensor)

    # 创建数据模块
    data_module = AHMDataModule(dataset, batch_size=cfg.training_params.batch_size)

    # 创建模型实例，传递配置字典
    train_model = SurrogateModel(config=cfg)

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

    # 获取保存模型的路径
    save_path = cfg.task_params.save_model_path

    # 检查路径是否存在，如果不存在则创建
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存模型的权重
    torch.save(train_model.state_dict(), save_path)

    # 测试模型
    trainer.test(train_model, datamodule=data_module, ckpt_path=best_model_path)

if __name__ == "__main__":
    main()
