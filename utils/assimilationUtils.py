import torch
import numpy as np

def initialize_parameters(eki, mean=0.0, std=1.0, low=-10, high=10):
    """
    Initialize parameter ensemble with values sampled from a truncated normal distribution.
    
    Args:
        eki: The ESMDA instance with configuration for ensemble size and parameter dimensions.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        low (float): Lower bound of the truncated range.
        high (float): Upper bound of the truncated range.
        
    Returns:
        torch.Tensor: Initialized parameter ensemble.
    """
    # Generate a normal distribution
    parameter_ensemble = torch.randn((eki.num_particles, *eki.parameter_dim), device=eki.device) * std + mean
    
    # Clamp values to the range [low, high]
    parameter_ensemble = torch.clamp(parameter_ensemble, low, high)
    
    return parameter_ensemble

def compute_parameter_posterior(prior, c_up, c_pp, r, r_matrix, h):
    return prior + torch.matmul(c_up, torch.linalg.solve(c_pp + 1 / h * r_matrix, r))

def perform_data_assimilation(eki, observation_operator, observations, parameter_ensemble):
    """
    Run the Ensemble Kalman Inversion and track intermediate outputs, errors, and variance.
    """
    all_outputs = []
    mse_list = []
    mae_list = []
    
    # Calculate initial parameter variance
    initial_parameter_variance = parameter_ensemble.var(dim=0).cpu().numpy()
    final_parameter_variance = None

    first_obs_prior = None  # Store the first obs_prior
    final_obs_prior = None  # Store the final obs_prior

    for i in range(eki.num_iterations):
        # Forward model computation
        output_prior = eki._compute_ensemble(parameter_ensemble)
        obs_prior = torch.stack([
            observation_operator(output_prior[j].to(eki.device))
            for j in range(eki.num_particles)
        ]).to(eki.device)

        # Store the first obs_prior
        if i == 0:
            first_obs_prior = obs_prior.clone()

        # Store outputs and compute errors
        all_outputs.append(obs_prior.mean(dim=0).cpu().numpy())
        mse = ((observations.cpu().numpy() - all_outputs[-1]) ** 2).mean()
        mae = np.abs(observations.cpu().numpy() - all_outputs[-1]).mean()
        mse_list.append(mse)
        mae_list.append(mae)

        # Convergence check based on relative MAE change
        if i > 0:  # Ensure there is a previous MAE to compare
            relative_mae_change = abs(mae_list[-1] - mae_list[-2]) / mae_list[-2]
            if relative_mae_change < 0.001:  # Change < 0.1%
                print(f"Convergence achieved with relative MAE change < 0.1% after {i + 1} iterations.")
                final_obs_prior = obs_prior.clone()
                break

        # Update parameters with EKI
        parameter_ensemble = update_parameters(eki, parameter_ensemble, obs_prior, observations)
        final_parameter_variance = parameter_ensemble.var(dim=0).cpu().numpy()

    # If the loop ends without convergence, set final_obs_prior to the last obs_prior
    if final_obs_prior is None:
        final_obs_prior = obs_prior.clone()

    return (
        all_outputs,
        mse_list,
        mae_list,
        initial_parameter_variance,
        final_parameter_variance,
        first_obs_prior.cpu().numpy(),
        final_obs_prior.cpu().numpy(),
    )

def generate_noise(observations, r_matrix, alpha_k, num_particles, device):
    """
    Generate observation error perturbation noise following N(0, alpha_k * R).
    
    Parameters:
    - observations: The observed data tensor.
    - R: The observation error covariance matrix (torch tensor).
    - alpha_k: The scaling factor for the iteration.
    - num_particles: The number of particles (ensemble members).
    - device: The device (CPU or GPU) to perform the calculations on.

    Returns:
    - noise: Perturbation noise matrix of size (num_particles, observation_dim).
    """
    # Scale the covariance matrix by alpha_k
    scaled_covariance = alpha_k * r_matrix

    # Define a multivariate normal distribution with zero mean and scaled covariance
    mvn_dist = torch.distributions.MultivariateNormal(
        torch.zeros(observations.shape[0], device=device), scaled_covariance
    )
    
    # Generate noise for each particle
    noise = mvn_dist.sample((num_particles,))
    
    return noise

def generate_r_matrix(observations, device):
    """
    Generate the observation error covariance matrix (R matrix).
    
    Parameters:
    - observations: The observed data tensor (used to determine the matrix size).
    - device: The device (CPU or GPU) to perform the calculations on.
    
    Returns:
    - r_matrix: Observation error covariance matrix (identity matrix in this example).
    """
    return torch.eye(observations.shape[0], device=device)

def update_parameters(eki, parameter_ensemble, obs_prior, observations):
    """Update parameters during data assimilation."""
    parameter_prior_mean = parameter_ensemble.mean(dim=0)
    obs_prior_mean = obs_prior.mean(dim=0)

    c_pp = torch.zeros((observations.shape[0], observations.shape[0]), device=eki.device)
    c_up = torch.zeros((eki.num_parameter_dofs, observations.shape[0]), device=eki.device)

    for j in range(eki.num_particles):
        c_pp += torch.outer(obs_prior[j, :] - obs_prior_mean, obs_prior[j, :] - obs_prior_mean)
        c_up += torch.outer(parameter_ensemble[j].view(-1) - parameter_prior_mean.view(-1), obs_prior[j, :] - obs_prior_mean)

    c_pp /= eki.num_particles
    c_up /= eki.num_particles

    # Generate observation error covariance matrix R
    R = generate_r_matrix(observations, eki.device)

    # Generate noise using the covariance matrix and scaling factor alpha_k
    alpha_k = 1 / eki.num_iterations  # Example scaling factor, can be adjusted
    noise = generate_noise(observations, R, alpha_k, eki.num_particles, eki.device)
    
    obs_perturbed = observations + noise
    r = obs_perturbed - obs_prior

    parameter_posterior = torch.zeros((eki.num_particles, *eki.parameter_dim), device=eki.device)
    for j in range(eki.num_particles):
        parameter_posterior[j] = compute_parameter_posterior(
            parameter_ensemble[j].view(-1), c_up, c_pp, r[j], R, eki.h
        ).view(eki.parameter_dim).to(eki.device)

    return parameter_posterior
