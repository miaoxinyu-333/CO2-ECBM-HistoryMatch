import torch
import os
from models.SurrogateModel import SurrogateModel
from models.AutoencoderKLWrapper import AutoencoderKLWrapper
from utils.dataUtils import remove_prefix
from models.ForwardModel import ForwardModel
from omegaconf import OmegaConf

class ForwardModelCreator:
    def __init__(self, device=None):
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Models for autoencoder and Fourier U-Net surrogate
        self.autoencoder_model = None
        self.surrogate_model = None

    def load_autoencoder(self, checkpoint_path):
        """Load the autoencoder model and its weights."""
        # 初始化 AutoencoderKLWrapper，传入学习率、KL权重等超参数
        self.autoencoder_model = AutoencoderKLWrapper()
        
        # 加载训练好的权重
        self.autoencoder_model.load_model_weights(checkpoint_path)
        self.autoencoder_model = self.autoencoder_model.autoencoder

    def load_surrogate(self):
        """Load Fourier U-Net surrogate model."""

        config_path = os.path.join("config", "SurrogateModel.yaml")
        task_name = 'task1'
        cfg = OmegaConf.load(config_path)

        cfg = cfg['tasks'][task_name]
        cfg_surrogate = cfg


        # Load the surrogate model structure
        self.surrogate_model = SurrogateModel(config=cfg_surrogate)

        # Load the surrogate model weights
        checkpoint_path = cfg_surrogate.task_params.save_model_path
        self.surrogate_model.load_model_weights(checkpoint_path)

        self.surrogate_model = self.surrogate_model.model

    def create_forward_model(self, autoencoder_checkpoint_path):
        """Instantiate and return the forward model."""
        # Load both models
        self.load_autoencoder(autoencoder_checkpoint_path)
        self.load_surrogate()

        # Instantiate the forward model class with the loaded models
        forward_model = ForwardModel(
            autoencoder_model=self.autoencoder_model,
            surrogate_model=self.surrogate_model
        )

        # Set the model to evaluation mode and move it to the appropriate device
        forward_model.eval()
        forward_model.to(self.device)

        return forward_model
