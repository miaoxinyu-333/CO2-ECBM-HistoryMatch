import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class AHMDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # 如果有数据下载或预处理的需求，可以在这里实现
        pass

    def setup(self, stage=None):
        # 设置随机种子，确保可重复性
        torch.manual_seed(42)
        
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        valid_size = int(0.2 * total_size)
        test_size = total_size - train_size - valid_size

        # 防止任何数据集分片大小为0
        if min(train_size, valid_size, test_size) <= 0:
            raise ValueError("数据集太小或分割比例不合理")

        # 根据阶段判断是否需要分割所有数据
        if stage == "fit" or stage is None:
            self.train_dataset, self.valid_dataset, _ = random_split(self.dataset, [train_size, valid_size, test_size])
        if stage == "test" or stage is None:
            _, _, self.test_dataset = random_split(self.dataset, [train_size, valid_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    # 获取训练数据集
    def get_train_dataset(self):
        return self.train_dataset

    # 获取验证数据集
    def get_val_dataset(self):
        return self.valid_dataset

    # 获取测试数据集
    def get_test_dataset(self):
        return self.test_dataset

