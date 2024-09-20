import torch
import os
import pickle
from models.AHMModel import AHMModel
from utils.dataUtils import remove_prefix
from models.ForwardModel import ForwardModel
from config.PCAModelConfig import PCAModelConfig
from config.FourierModelConfig import load_config
from omegaconf import OmegaConf

class ForwardModelCreator:
    def __init__(self, device=None):
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Config paths
        self.config_path_fourier = os.path.join("config", "FourierUNet.yaml")
        self.config_path_pca = os.path.join("config", "pca.yaml")

        # Models for 'per' and 'por' PCA, Autoencoder, and Surrogate
        self.pca_model_per = None
        self.pca_model_por = None
        self.autoencoder_per = None
        self.autoencoder_por = None
        self.surrogate_model = None

    def load_pca_models(self):
        """Load PCA models for permeability and porosity."""
        config_pca = PCAModelConfig(self.config_path_pca)
        pca_model_path_per = config_pca.save_model_path_per
        pca_model_path_por = config_pca.save_model_path_por

        # Load PCA models
        with open(pca_model_path_per, 'rb') as f_per:
            self.pca_model_per = pickle.load(f_per)

        with open(pca_model_path_por, 'rb') as f_por:
            self.pca_model_por = pickle.load(f_por)

    def load_autoencoder_models(self):
        """Load autoencoder models for permeability and porosity."""
        task_per = "task2"
        task_por = "task3"
        cfg_per = load_config(self.config_path_fourier,task_per)
        cfg_por = load_config(self.config_path_fourier,task_por)
        prefix = "model."

        # Load the model structure
        self.autoencoder_per = AHMModel(config=cfg_per.to_dict())
        self.autoencoder_por = AHMModel(config=cfg_por.to_dict())

        # Load the model weights
        autoencoder_checkpoint_path_per = cfg_per.task_params.save_model_path
        autoencoder_checkpoint_path_por = cfg_por.task_params.save_model_path

        # Load model weights from .pth files
        autoencoder_checkpoint_per = torch.load(autoencoder_checkpoint_path_per, map_location=self.device)
        autoencoder_checkpoint_por = torch.load(autoencoder_checkpoint_path_por, map_location=self.device)

        autoencoder_state_dict_per = remove_prefix(autoencoder_checkpoint_per, prefix)
        autoencoder_state_dict_por = remove_prefix(autoencoder_checkpoint_por, prefix)

        self.autoencoder_per.model.load_state_dict(autoencoder_state_dict_per)
        self.autoencoder_por.model.load_state_dict(autoencoder_state_dict_por)

    def load_surrogate_model(self):
        """Load the surrogate (UNet) model."""
        task_surrogate = "task1"
        cfg_surogate = load_config(self.config_path_fourier, task_surrogate)
        prefix = "model."

        # Load surrogate model structure
        self.surrogate_model = AHMModel(config=cfg_surogate.to_dict())

        # Load the surrogate model weights
        unet_checkpoint_path = cfg_surogate.task_params.save_model_path
        unet_checkpoint = torch.load(unet_checkpoint_path, map_location=self.device)
        unet_state_dict = remove_prefix(unet_checkpoint, prefix)

        # Load the state dictionary into the model
        self.surrogate_model.model.load_state_dict(unet_state_dict)

    def create_forward_model(self):
        """Instantiate and return the forward model."""
        # Load all necessary models
        self.load_pca_models()
        self.load_autoencoder_models()
        self.load_surrogate_model()

        # Instantiate the forward model class with the loaded models
        forward_model = ForwardModel(
            pca_model_per=self.pca_model_per,
            pca_model_por=self.pca_model_por,
            autoencoder_per=self.autoencoder_per,
            autoencoder_por=self.autoencoder_por,
            surrogate_model=self.surrogate_model
        )

        # Set the model to evaluation mode and move it to the appropriate device
        forward_model.eval()
        forward_model.to(self.device)

        return forward_model