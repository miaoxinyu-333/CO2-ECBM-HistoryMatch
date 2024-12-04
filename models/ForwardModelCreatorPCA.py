import torch
import os
import pickle
from models.SurrogateModel import SurrogateModel
from models.ForwardModelPCA import ForwardModelPCA
from omegaconf import OmegaConf

class ForwardModelCreatorPCA:
    def __init__(self, device=None):
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Models for PCA and surrogate model
        self.pca_model = None
        self.surrogate_model = None

    def load_pca(self, pca_file_path):
        """Load the PCA model from a file."""
        with open(pca_file_path, 'rb') as f:
            self.pca_model = pickle.load(f)
        print(f"PCA model loaded from {pca_file_path}")

    def load_surrogate(self):
        """Load U-Net surrogate model."""
        config_path = os.path.join("config", "SurrogateModel.yaml")
        task_name = 'task2'
        cfg = OmegaConf.load(config_path)
        cfg = cfg['tasks'][task_name]
        cfg_surrogate = cfg

        # Load the surrogate model structure
        self.surrogate_model = SurrogateModel(config=cfg_surrogate)

        # Load the surrogate model weights
        checkpoint_path = cfg_surrogate.task_params.save_model_path
        self.surrogate_model.load_model_weights(checkpoint_path)

        self.surrogate_model = self.surrogate_model.model
        print(f"Surrogate model loaded from {checkpoint_path}")

    def create_forward_model(self, pca_path):
        """
        Instantiate and return the forward model using PCA.

        Args:
            pca_path (str): The path to the PCA model file.
        """
        # Load PCA model
        self.load_pca(pca_path)

        # Load surrogate model
        self.load_surrogate()

        # Instantiate the forward model class with the loaded models
        forward_model = ForwardModelPCA(
            pca_model=self.pca_model,  # Pass PCA model
            surrogate_model=self.surrogate_model
        )

        # Set the model to evaluation mode and move it to the appropriate device
        forward_model.eval()
        forward_model.to(self.device)

        return forward_model
