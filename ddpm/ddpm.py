import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPM(nn.Module):

    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.network = network.to(device)

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # We add the gaussian noise

        sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        s1 = sqrt_alphas_cumprod[timesteps].to(device)
        s2 = sqrt_one_minus_alphas_cumprod[timesteps].to(device)
        s1 = s1.reshape(-1, 1).unsqueeze(2).unsqueeze(3)
        s2 = s2.reshape(-1, 1).unsqueeze(2).unsqueeze(3)

        return s1 * x_start + s2 * x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added

        return self.network.forward(x, t)

    def reconstruct_x0(self, x_t, t, noise):
        # We reconstruct the origin image with the noise predicted

        s1 = torch.sqrt(1 / self.alphas_cumprod)[t].to(device)
        s2 = torch.sqrt(1 / self.alphas_cumprod - 1)[t].to(device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        # The equiation (7) in the DDPM paper
        # To calculate the mean posterior of the x_{t-1}|x_{t}， x_{0}

        posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(device)
        posterior_mean_coef2= ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)).to(device)
        s1 = posterior_mean_coef1[t].to(device)
        s2 = posterior_mean_coef2[t].to(device)
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t

        return mu

    def get_variance(self, t):
        # The equiation (7) in the DDPM paper
        # To calculate the posterior coefficient of the covariance matrix of the x_{t-1}|x_{t}， x_{0}

        if t == 0:
            return 0

        variance = (self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])).to(device)
        variance = variance.clip(1e-20)

        return variance

    def step(self, model_output, timestep, sample):
        # reverse step x_{t}->x_{t-1}
        
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output).to(device)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t).to(device)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(device)
            variance = ((self.get_variance(t) ** 0.5) * noise).to(device)

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
