import torch
import torch.nn as nn
from tqdm import tqdm

def compute_parameter_posterior(prior, c_up, c_pp, r, r_matrix, h):
    return prior + torch.matmul(c_up, torch.linalg.solve(c_pp + 1 / h * r_matrix, r))

class ESMDA:
    def __init__(
        self,
        forward_model: nn.Module,
        num_particles: int = 100,
        num_iterations: int = 100,
        parameter_dim: tuple = (2, 300),  # Updated to match new input shape
        device: str = 'cuda'
    ) -> None:
        self.device = device
        self.forward_model = forward_model.to(device)
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.h = 1 / self.num_iterations
        self.parameter_dim = parameter_dim
        self.num_parameter_dofs = parameter_dim[0] * parameter_dim[1]  # 2 * 300 = 600
        self.batch_size = 25  # Adjust batch size as needed
        self.output_dim = (self.num_particles, 12, 1, 32, 32)  # Maintain the output shape

    def _compute_ensemble(self, parameters: torch.Tensor) -> torch.Tensor:
        parameters = parameters.reshape((self.num_particles, *self.parameter_dim)).to(self.device)
        output_shape = (self.num_particles, 12, 1, 32, 32)
        model_output = torch.zeros(output_shape, device=self.device)

        with torch.no_grad():
            for i in range(0, self.num_particles, self.batch_size):
                parameters_batch = parameters[i:i + self.batch_size].to(self.device)
                model_output[i:i + self.batch_size] = self.forward_model(
                    parameters_batch
                ).to(self.device)

        return model_output

    def solve(self, observation_operator: callable, observations: torch.Tensor, noise_std: float = 0.1):
        self.forward_model.eval()
        observations = observations.to(self.device)
        num_observations = observations.shape[0]
        r_matrix = 0.005 * torch.eye(observations.shape[0], device=self.device)
        
        # Initialize parameter ensemble with the new shape
        parameter_ensemble = torch.rand((self.num_particles, *self.parameter_dim), device=self.device)

        pbar = tqdm(range(self.num_iterations), total=self.num_iterations, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for _ in pbar:
            output_prior = self._compute_ensemble(parameter_ensemble)
            obs_prior = torch.stack([observation_operator(output_prior[j].to(self.device)) for j in range(self.num_particles)]).to(self.device)

            parameter_prior_mean = parameter_ensemble.mean(dim=0)
            obs_prior_mean = obs_prior.mean(dim=0)

            c_pp = torch.zeros((num_observations, num_observations), device=self.device)
            c_up = torch.zeros((self.num_parameter_dofs, num_observations), device=self.device)
            for j in range(self.num_particles):
                c_pp += torch.outer(obs_prior[j, :] - obs_prior_mean, obs_prior[j, :] - obs_prior_mean)
                c_up += torch.outer(parameter_ensemble[j].view(-1) - parameter_prior_mean.view(-1), obs_prior[j, :] - obs_prior_mean)
            c_pp /= self.num_particles
            c_up /= self.num_particles

            noise = torch.normal(mean=0.0, std=noise_std, size=(self.num_particles, num_observations), device=self.device)
            obs_perturbed = observations + noise
            r = obs_perturbed - obs_prior

            parameter_posterior = torch.zeros((self.num_particles, *self.parameter_dim), device=self.device)
            for j in range(self.num_particles):
                parameter_posterior[j] = compute_parameter_posterior(parameter_ensemble[j].view(-1), c_up, c_pp, r[j], r_matrix, self.h).view(self.parameter_dim).to(self.device)

            parameter_ensemble = parameter_posterior

        output_posterior = self._compute_ensemble(parameter_ensemble)
        return parameter_ensemble.cpu(), output_posterior.cpu()
