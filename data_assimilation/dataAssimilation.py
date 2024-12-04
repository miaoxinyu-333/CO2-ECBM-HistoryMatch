import torch
import os
import sys
import numpy as np

# 获取项目根目录
def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)


def main():
    # Setup
    setup_project_root()
    from utils.dataUtils import get_obsdata
    from utils.dataUtils import observation_operator
    from modules.ESMDA import ESMDA
    from models.ForwardModelCreator import ForwardModelCreator
    from utils.assimilationUtils import initialize_parameters
    from utils.assimilationUtils import perform_data_assimilation
    from utils.plotUtils import plot_concentration
    from utils.plotUtils import set_plotting_params

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    autoencoder_checkpoint_path = "logs/AutoencoderKL_PTH/best_autoencoder_model.pth"
    co2_file_path = "dataSet/surrogate/dataset_co2_4983.h5"

    """Load the forward model and observation data."""
    forward_model_creator = ForwardModelCreator()
    forward_model = forward_model_creator.create_forward_model(autoencoder_checkpoint_path)
    observations = get_obsdata(co2_file_path).to(device)

    """Initialize Ensemble Kalman Inversion."""
    eki = ESMDA(
        forward_model=forward_model,
        num_particles=100,
        num_iterations=100,
        parameter_dim=(4,8,8),
        device=device
    )

    # Initialize parameters
    parameter_ensemble = initialize_parameters(eki)

    # Run data assimilation and get results
    all_outputs, mse_list, mae_list, initial_variance, final_variance, output_prior_initial, output_prior_final = perform_data_assimilation(
        eki, observation_operator, observations, parameter_ensemble)
    

    minimum_prediction = np.array([
    1285.5818, 1586.4941, 1727.458, 1834.1538, 1919.1346, 1972.3094,
    2013.9987, 2054.2048, 2077.6807, 2095.0388, 2121.5557, 2169.4663
    ])

    # Maximum Prediction
    maximum_prediction = np.array([
        2165.6245, 2394.4797, 2538.8801, 2610.7642, 2676.7249, 2714.4897,
        2733.736, 2763.2712, 2786.3281, 2793.3984, 2799.5088, 2814.2512
    ])

    observations = observations.cpu().numpy()
    print(minimum_prediction.shape)
    print(maximum_prediction.shape)
    print(observations.shape)

    set_plotting_params()
    
    plot_concentration(output_prior_initial, output_prior_final, minimum_prediction, maximum_prediction, observations)

    # Calculate and print average variance
    initial_variance_mean = np.mean(initial_variance)
    final_variance_mean = np.mean(final_variance)
    print(f"Average Initial Variance: {initial_variance_mean:.6f}")
    print(f"Average Final Variance: {final_variance_mean:.6f}")

    # Display MSE and MAE values
    print(f"Initial MSE: {mse_list[0]:.6f}, Final MSE: {mse_list[-1]:.6f}")
    print(f"Initial MAE: {mae_list[0]:.6f}, Final MAE: {mae_list[-1]:.6f}")

if __name__ == "__main__":
    main()
