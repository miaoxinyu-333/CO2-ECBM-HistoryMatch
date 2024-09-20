import torch
import os
import sys

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
    from modules.ForwardModelCreator import ForwardModelCreator
    from utils.assimilationUtils import initialize_parameters
    from utils.assimilationUtils import perform_data_assimilation
    from utils.plotUtils import plot_assimilation_process
    from utils.plotUtils import plot_errors_over_iterations
    from utils.plotUtils import plot_parameter_variance
    from utils.plotUtils import set_plotting_params
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """Load the forward model and observation data."""
    forward_model_creator = ForwardModelCreator()
    forward_model = forward_model_creator.create_forward_model()
    observations = get_obsdata().to(device)

    """Initialize Ensemble Kalman Inversion."""
    eki = ESMDA(
        forward_model=forward_model,
        num_particles=100,
        num_iterations=25,
        parameter_dim=(2, 300),
        device=device
    )

    # Initialize parameters
    parameter_ensemble = initialize_parameters(eki)

    # Run data assimilation and get results
    all_outputs, mse_list, mae_list, initial_variance, final_variance = perform_data_assimilation(
        eki, observation_operator, observations, parameter_ensemble)

    set_plotting_params()

    # Plot results
    save_path = 'logs/images/assimilation'
    plot_parameter_variance(initial_variance, final_variance, save_dir=save_path)
    plot_errors_over_iterations(mse_list, mae_list, eki.num_iterations, save_dir=save_path)
    plot_assimilation_process(np.array(all_outputs), observations.cpu().numpy(), save_dir=save_path)

    # Final error
    final_mse = mse_list[-1]
    final_mae = mae_list[-1]
    print(f"Final MSE: {final_mse}, Final MAE: {final_mae}")

if __name__ == "__main__":
    main()
