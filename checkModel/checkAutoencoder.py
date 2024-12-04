import os
import sys
from torchinfo import summary

def setup_project_root():
    """
    Adds the project root to sys.path to ensure correct module imports.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def main():
    """
    Main function to inspect the AutoencoderKLWrapper model using torchinfo.
    """
    setup_project_root()

    import torch
    from models.AutoencoderKLWrapper import AutoencoderKLWrapper
    from utils.dataUtils import load_per_por_from_h5

    # Parameters for model initialization and input shape
    params = {
        "data_set_path": "dataSet/surrogate/dataset_per_por_4983.h5",
        "batch_size": 32,
        "learning_rate": 1e-3,
        "kl_weight": 1e-7,
        "weight_decay": 0.0,
    }

    # Initialize the AutoencoderKLWrapper model
    model = AutoencoderKLWrapper(
        lr=params["learning_rate"],
        kl_weight=params["kl_weight"],
        weight_decay=params["weight_decay"]
    )

    # Define input shape
    input_channels = 2  # Permeability + Porosity
    height, width = 64, 64  # Assuming 64x64 grid
    batch_size = params["batch_size"]

    # Adjust the input shape to match the time dimension
    input_shape = (batch_size, input_channels, height, width)

    # Inspect the model using torchinfo
    print("Inspecting AutoencoderKLWrapper Model:")
    summary(
        model,
        input_size=input_shape,  # Batch size, Time, Channels, Height, Width
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=5  # Adjust depth for detailed or summary view
    )

if __name__ == "__main__":
    main()
