import pytorch_lightning as pl
import torch

class ForwardModelPCA(pl.LightningModule):
    def __init__(self, pca_model, surrogate_model, device='cuda'):
        super(ForwardModelPCA, self).__init__()

        # Assign the models to instance variables
        self.pca_model = pca_model
        self.surrogate_model = surrogate_model
        
        # Assign the device to an instance variable
        self.model_device = device
        
        # Move surrogate model to the specified device
        self.surrogate_model.to(self.model_device)

        # Optional: Print or log model device information
        print(f"Surrogate model has been moved to device: {self.model_device}")

    def forward(self, x):
        """
        Perform a forward pass using PCA for encoding/decoding and the surrogate model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor after surrogate model prediction.
        """
        # Ensure input tensor is on the correct device
        x = x.to(self.model_device)

        # Extract dimensions for reconstruction
        batch_size, n_channels, height, width = x.shape

        # Flatten input tensor and move it to CPU for PCA
        x_flattened = x.cpu().numpy()  # Convert to numpy for PCA operations
        reduced_data = self.pca_model.transform(x_flattened)  # Call transform

        # Decode back using inverse_transform
        reconstructed_tensor = self.pca_model.inverse_transform(reduced_data, n_channels, height, width)

        reconstructed_tensor = reconstructed_tensor.unsqueeze(1)
        reconstructed_tensor = reconstructed_tensor.to(self.model_device)

        # Use the surrogate model for prediction
        x_co2 = self.surrogate_model(reconstructed_tensor)

        return x_co2
