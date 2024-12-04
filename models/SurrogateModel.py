import pytorch_lightning as pl
import torch
from torch.optim import Adam
from models.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from models.lr_scheduler import LinearWarmupCosineAnnealingLR
from evaluation.eval_metric import normalized_root_mean_squared_error
from evaluation.eval_metric import r2_score_pytorch
from evaluation.eval_metric import ssim_index
from evaluation.eval_metric import mean_squared_error

class SurrogateModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = ModelFactory.create_model(**config['model_params'])
        
        # 获取损失函数
        self.criterion = LossFactory.get_loss_function(config['loss_function'])
        
        # 获取优化器和调度器参数
        self.optimizer_params = config['optimizer_params']
        self.scheduler_params = config['scheduler_params']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)
        
        # Calculate additional metrics
        nrmse = normalized_root_mean_squared_error(targets, outputs)
        r2 = r2_score_pytorch(targets, outputs)
        ssim_val = ssim_index(targets, outputs)
        mse = mean_squared_error(targets, outputs)  # Add MSE calculation
        
        # Logging
        self.log('val_nrmse', nrmse)
        self.log('val_r2', r2)
        self.log('val_ssim', ssim_val)
        self.log('val_mse', mse)  # Log MSE

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('test_loss', loss)
        
        # Calculate and log metrics
        nrmse = normalized_root_mean_squared_error(targets, outputs)
        r2 = r2_score_pytorch(targets, outputs)
        ssim_val = ssim_index(targets, outputs)
        mse = mean_squared_error(targets, outputs)  # Add MSE calculation
        
        self.log('test_nrmse', nrmse)
        self.log('test_r2', r2)
        self.log('test_ssim', ssim_val)
        self.log('test_mse', mse)  # Log MSE

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), **self.optimizer_params)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **self.scheduler_params)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def load_model_weights(self, checkpoint_path):
        """加载训练好的模型权重，去掉前缀"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 如果 checkpoint 中有 "state_dict"，获取它
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # 去掉前缀 "autoencoder." 
        prefix = 'model.'
        checkpoint = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}

        # 加载去掉前缀后的权重
        self.model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {checkpoint_path}")